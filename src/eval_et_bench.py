import torch, argparse, json, os, re, tqdm, copy
from transformers import Qwen2VLProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

FRAME_ARGS = {
    'min_pixels': 16 * 28 * 28,
    'total_pixels': 3584 * 28 * 28,
}

TEMPLATE = (
    '{} First, output reasoning process in <think> </think> tags. The reasoning process must REFER TO SPECIFIC TIMESTAMPS TO TELL WHERE YOU GET THE INFORMATION FROM THE VIDEO. '
    'Then summarize your reasoning process above and output your answer within <answer> </answer> tags. Make sure to follow the format requirements. '
    'Your output format should be like \"<think>...</think><answer>...</answer>\".'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--video_path', type=str, default='./videos/et_bench_processed')
    parser.add_argument('--data_path', type=str, default='./data/eval/et_bench/et_bench.json')
    parser.add_argument('--subset_path', type=str, default='./data/eval/et_bench/subset.json')
    parser.add_argument('--output_path', type=str, default='./eval_et_bench.json')
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--batch_num', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.data_path, 'r') as fp:
        data_raw = json.load(fp)
    if args.subset:
        data = list()
        with open(args.subset_path, 'r') as fp:
            subset = json.load(fp)
            for entry in data_raw:
                if entry['source'] in subset[entry['task']].keys():
                    if entry['idx'] in subset[entry['task']][entry['source']]:
                        data.append(entry)
    else:
        data = data_raw
    model = LLM(model=args.ckpt_path, max_model_len=128000, max_num_seqs=args.batch_num)
    processor = Qwen2VLProcessor.from_pretrained(args.ckpt_path)

    batched_data = list()
    for entry in data:
        if len(batched_data) == 0 or len(batched_data[-1]) >= args.batch_num:
            batched_data.append(list())
        batched_data[-1].append(entry)

    entry_ptr = -1
    video_cache = ('', None, None)
    for batch in tqdm.tqdm(batched_data):
        conversations, videos_fps = list(), list()
        # make input
        for entry in batch:
            video_path = os.path.join(args.video_path, entry['video'])
            if video_cache[0] == video_path:
                frame_files, fps = copy.deepcopy(video_cache[1]), video_cache[2]
            else:
                vid_input = {'type': 'video', 'video': video_path}
                for k, v in FRAME_ARGS.items():
                    vid_input[k] = v 
                _, frame_files, fps = process_vision_info([{'content': [vid_input]}], return_video_kwargs=True)
                frame_files, fps = frame_files[0], fps['fps'][0]
                video_cache = (video_path, frame_files, fps)
            conversations.append([{
                'role': 'user',
                'content': [
                    { 'type': 'video', 'video': frame_files },
                    { 'type': 'text', 'text': TEMPLATE.format(entry['q']) },
                ],
            }])
            videos_fps.append(fps)
        # inference
        with torch.no_grad():
            text = [processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
            inp = [{'prompt': item, 'multi_modal_data': {'video': conversation[-1]['content'][0]['video']}, 'mm_processor_kwargs': {'fps': fps}} for item, conversation, fps in zip(text, conversations, videos_fps)]
            generation_config = SamplingParams(top_p=1, top_k=-1, temperature=0, max_tokens=512, stop_token_ids=None)
            out = model.generate(inp, sampling_params=generation_config)
            outputs = [o.outputs[0].text for o in out]
        # save answer
        for conversation, output in zip(conversations, outputs):
            conversation.append({'role': 'assistant', 'content': [{'type': 'text', 'text': output}]})
            for msg in conversation:
                msg['content'] = ''.join([(item['text'] if (item['type'] == 'text') else '<video>') for item in msg['content']])
            try:
                assert len(re.findall(f'<answer>', output, re.DOTALL)) == 1
                assert len(re.findall(f'</answer>', output, re.DOTALL)) == 1
                output = re.search(f'<answer>(.*)</answer>', output, re.DOTALL).group(1).strip()
            except:
                output = ''
            entry_ptr += 1
            data[entry_ptr]['a'] = output
            data[entry_ptr]['c'] = [(msg['role'] + ': ' + msg['content']) for msg in conversation]

    with open(args.output_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    main()
