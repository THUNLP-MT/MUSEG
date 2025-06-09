import torch, nncore.ops, argparse, json, os, re, tqdm, copy
from transformers import Qwen2VLProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

FRAME_ARGS = {
    'min_pixels': 16 * 28 * 28,
    'total_pixels': 3584 * 28 * 28,
}

TEMPLATE = (
    '{} First, output reasoning process in <think> </think> tags. The reasoning process must REFER TO SPECIFIC TIMESTAMPS TO TELL WHERE YOU GET THE INFORMATION FROM THE VIDEO. '
    'Then summarize your reasoning process above and output selected segments like \"<answer>X.XX-X.XX</answer>\", where \"X\" denotes arabic numbers. If there are multiple segments, separate them with spaces like \"<answer>X.XX-X.XX X.XX-X.XX</answer>\". '
    'Your output format should be like \"<think>...</think><answer>...</answer>\".'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--video_path', type=str, default='./videos')
    parser.add_argument('--data_path', type=str, default='./data/eval/grounding/charades_sta.json')
    parser.add_argument('--output_path', type=str, default='./eval_charades_sta.json')
    parser.add_argument('--eval_method', type=str, default='tvg')
    parser.add_argument('--batch_num', type=int, default=10)
    return parser.parse_args()


def eval(data_raw, eval_type):
    data, thrs = list(), [0.1, 0.3, 0.5, 0.7]
    for entry in data_raw:
        segs, output = entry[0], entry[1]
        try:
            segs_sel = re.findall(r'\d+\.*\d*', output)
            assert len(segs_sel) % 2 == 0
            segs_sel = [[float(segs_sel[i]), float(segs_sel[i + 1])] for i in range(0, len(segs_sel), 2)]
            for seg_sel in segs_sel:
                assert seg_sel[0] < seg_sel[1]
        except:
            segs_sel = list()
        data.append([segs, segs_sel])
    m_iou, res_f1_single, res_f1_multi = 0, [0] * len(thrs), [0] * len(thrs)
    for (segs, segs_sel) in data:
        if len(segs_sel) != 0:
            iou_single = nncore.ops.temporal_iou(torch.Tensor([segs[0]]), torch.Tensor([segs_sel[0]]))
            iou_multi = nncore.ops.temporal_iou(torch.Tensor(segs), torch.Tensor(segs_sel))
            m_iou += iou_single.item()
            for i, thr in enumerate(thrs):
                res_f1_single[i] += int(iou_single.item() >= thr)
                if iou_multi.max() < thr:
                    continue
                else:
                    rec = (iou_multi.amax(dim=1) >= thr).float().mean().item()
                    prc = (iou_multi.amax(dim=0) >= thr).float().mean().item()
                    if prc > 0 and rec > 0:
                        res_f1_multi[i] += 2 * prc * rec / (prc + rec)

    stat = { 'tot': len(data) }
    if eval_type == 'tvg':
        stat['m_iou'] = m_iou / len(data)
        for i, thr in enumerate(thrs):
            stat[f'f1_single_{thr}'] = res_f1_single[i] / len(data)
        stat['mean_f1_single'] = sum(res_f1_single) / len(data) / len(res_f1_single)
    elif eval_type == 'tal':
        for i, thr in enumerate(thrs):
            stat[f'f1_multi_{thr}'] = res_f1_multi[i] / len(data)
        stat['mean_f1_multi'] = sum(res_f1_multi) / len(data) / len(res_f1_multi)
    else:
        raise NotImplementedError()
    return stat


def main():
    args = parse_args()
    with open(args.data_path, 'r') as fp:
        data = json.load(fp)
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
                    { 'type': 'text', 'text': TEMPLATE.format(entry['question']) },
                ],
            }])
            videos_fps.append(fps)
        # inference
        with torch.no_grad():
            text = [processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True) for conversation in conversations]
            inp = [{'prompt': item, 'multi_modal_data': {'video': conversation[-1]['content'][0]['video']}, 'mm_processor_kwargs': {'fps': fps}} for item, conversation, fps in zip(text, conversations, videos_fps)]
            generation_config = SamplingParams(temperature=0, max_tokens=512)
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
            data[entry_ptr]['result'] = {
                'output': output,
                'conversation': [(msg['role'] + ': ' + msg['content']) for msg in conversation]
            }

    data.append(eval([[entry['groundtruth'], entry['result']['output']] for entry in data], args.eval_method))
    with open(args.output_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    main()
