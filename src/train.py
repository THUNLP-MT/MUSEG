import os, copy, re
from dataclasses import dataclass, field
from typing import Optional, Union

from datasets import Dataset, DatasetDict
from trainer import GRPOVLLMTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ['format', 'segment_matching'],
        metadata={'help': 'List of reward functions.'},
    )
    question_template: Optional[str] = field(
        default='frame_selection_thinking',
        metadata={'help': 'Question template'}
    )
    max_pixels: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of pixels for the image'},
    )
    min_pixels: Optional[int] = field(
        default=None,
        metadata={'help': 'Minimum number of pixels for the image'},
    )
    max_frames: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of frames'},
    )
    nframes: Optional[Union[int, str]] = field(
        default=None,
        metadata={'help': 'Number of frames'},
    )
    total_pixels: Optional[int] = field(
        default=None,
        metadata={'help': 'total pixels'},
    )
    jsonl_path: Optional[str] = field(
        default=None,
        metadata={'help': 'json file path'},
    )


def _reward_format(include_timestamp_reward, completions, problem, solution, **kwargs):
    pattern = r'^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$'
    completion_contents = [completion[0]['content'] for completion in completions]
    rewards = list()
    for prob, content, task, idx in zip(problem, completion_contents, kwargs['task'], kwargs['idx']):
        exception = None
        try:
            reward = 0.0
            assert len(re.findall(r'<think>', content, re.DOTALL)) == 1
            assert len(re.findall(r'</think>', content, re.DOTALL)) == 1
            assert len(re.findall(r'<answer>', content, re.DOTALL)) == 1
            assert len(re.findall(r'</answer>', content, re.DOTALL)) == 1
            content_match = re.search(pattern, content, re.DOTALL).groups()
            thinking, segment_str = content_match[0], content_match[1]
            reward = 1.0
            # get time segments
            timestamps = list()
            for item in segment_str.strip().split():
                try:
                    t = item.split('-')
                    timestamps.extend([float(t[0]), float(t[1])])
                except:
                    reward = 0.2
                    break
            # timestamp reward
            if include_timestamp_reward == True and reward == 1.0:
                for timestamp in timestamps:
                    timestamp_match = re.findall(str(timestamp), thinking, re.DOTALL)
                    if len(timestamp_match) < 1 or len(timestamp_match) > 10:
                        reward = 0.6
                        break
        except Exception as e:
            exception = e
            pass
        rewards.append(reward)
        if os.getenv('DEBUG_MODE') == 'true':
            log_path = os.getenv('LOG_PATH')
            content_output = re.sub(r'[^\x00-\x7F]+', '', content)
            with open(log_path, 'a') as f:
                f.write(f'[### SAMPLE ###]\n')
                f.write(f'[### INFO ###] {idx} | FORMAT | {task} [### /INFO ###]\n')
                f.write(f'[### PROBLEM ###]\n{prob}\n[### /PROBLEM ###]\n')
                f.write(f'[### OUTPUT ###]\n{content_output}\n[### /OUTPUT ###]\n')
                f.write(f'[### REWARD ###]\n{reward}\n[### /REWARD ###]\n')
                f.write(f'[### EXCEPTION ###]\n{exception}\n[### /EXCEPTION ###]\n')
                f.write(f'[### /SAMPLE ###]\n')
    return rewards


def reward_format(completions, problem, solution, **kwargs):
    return _reward_format(True, completions, problem, solution, **kwargs)


def reward_format_n_timestamp(completions, problem, solution, **kwargs):
    return _reward_format(False, completions, problem, solution, **kwargs)


def reward_global_matching(completions, problem, solution, **kwargs):
    pattern = r'<answer>(.*?)</answer>'
    completion_contents = [completion[0]['content'] for completion in completions]
    rewards = list()
    for prob, content, gts, task, idx in zip(problem, completion_contents, solution, kwargs['task'], kwargs['idx']):
        exception = None
        try:
            assert len(re.findall(r'<answer>', content, re.DOTALL)) == 1
            assert len(re.findall(r'</answer>', content, re.DOTALL)) == 1
            content_match = re.search(pattern, content, re.DOTALL).group(1)
            # get time segments
            segments = list()
            for item in content_match.strip().split():
                t = item.split('-')
                try:
                    st, ed = float(t[0]), float(t[1])
                except:
                    assert False
                assert st >= 0 and ed >= 0
                assert st < ed
                segments.append([st, ed])
            segments = sorted(segments, key=lambda s: s[0])
            while True:
                no_overlap = True
                for i in range(len(segments) - 1):
                    if segments[i][1] >= segments[i + 1][0]:
                        segments[i][0] = min(segments[i][0], segments[i + 1][0])
                        segments[i][1] = max(segments[i][1], segments[i + 1][1])
                        segments[i + 1] = None
                        no_overlap = False
                        break
                if no_overlap == False:
                    segments = [item for item in segments if item is not None]
                else:
                    break
            if isinstance(gts[0], int):
                gts = [gts]
            all_gt, all_sel, overlap = sum([(item[1] - item[0]) for item in gts]), sum([(item[1] - item[0]) for item in segments]), 0
            for gt in gts:
                for segment in segments:
                    overlap += max(0, min(segment[1], gt[1]) - max(segment[0], gt[0]))
            score_precision, score_recall = overlap / (all_sel + 1e-6), overlap / (all_gt + 1e-6)
            reward = 2 * score_precision * score_recall / (score_precision + score_recall + 1e-6)
        except Exception as e:
            exception = e
            reward = 0.0
        rewards.append(reward)
        if os.getenv('DEBUG_MODE') == 'true':
            log_path = os.getenv('LOG_PATH')
            content_output = re.sub(r'[^\x00-\x7F]+', '', content)
            with open(log_path, 'a') as f:
                f.write(f'[### SAMPLE ###]\n')
                f.write(f'[### INFO ###] {idx} | GLOBAL MATCHING | {task} [### /INFO ###]\n')
                f.write(f'[### PROBLEM ###]\n{prob}\n[### /PROBLEM ###]\n')
                f.write(f'[### GROUNDTRUTH ###]\n{gts}\n[### /GROUNDTRUTH ###]\n')
                f.write(f'[### OUTPUT ###]\n{content_output}\n[### /OUTPUT ###]\n')
                f.write(f'[### REWARD ###]\n{reward}\n[### /REWARD ###]\n')
                f.write(f'[### EXCEPTION ###]\n{exception}\n[### /EXCEPTION ###]\n')
                f.write(f'[### /SAMPLE ###]\n')
    return rewards


def reward_local_matching(completions, problem, solution, **kwargs):
    pattern = r'<answer>(.*?)</answer>'
    completion_contents = [completion[0]['content'] for completion in completions]
    rewards = list()
    for prob, content, gts, task, idx in zip(problem, completion_contents, solution, kwargs['task'], kwargs['idx']):
        exception = None
        try:
            assert len(re.findall(r'<answer>', content, re.DOTALL)) == 1
            assert len(re.findall(r'</answer>', content, re.DOTALL)) == 1
            segments = list()
            content_match = re.search(pattern, content, re.DOTALL).group(1)
            # get time segments
            for item in content_match.strip().split():
                t = item.split('-')
                try:
                    st, ed = float(t[0]), float(t[1])
                except:
                    assert False
                assert st >= 0 and ed >= 0
                assert st < ed
                segments.append([st, ed])

            gts = copy.deepcopy(gts)
            scores = list()
            if len(gts) < len(segments):
                gts.extend([[1e9, 1e9 + 1]] * (len(segments) - len(gts)))
            elif len(segments) < len(gts):
                segments.extend([[1e9, 1e9 + 1]] * (len(gts) - len(segments)))
            for gt, segment in zip(gts, segments):
                intersection = max(0, min(gt[1], segment[1]) - max(gt[0], segment[0]))
                union = gt[1] - gt[0] + segment[1] - segment[0] - intersection
                enclosure = max(gt[1], segment[1]) - min(gt[0], segment[0])
                scores.append((1 + intersection / union - (enclosure - union) / enclosure) / 2)
            reward = sum(scores) / len(scores)
        except Exception as e:
            segments = list()
            exception = e
            reward = 0.0
        rewards.append(reward)
        if os.getenv('DEBUG_MODE') == 'true':
            log_path = os.getenv('LOG_PATH')
            content_output = re.sub(r'[^\x00-\x7F]+', '', content)
            with open(log_path, 'a') as f:
                f.write(f'[### SAMPLE ###]\n')
                f.write(f'[### INFO ###] {idx} | LOCAL MATCHING | {task} [### /INFO ###]\n')
                f.write(f'[### PROBLEM ###]\n{prob}\n[### /PROBLEM ###]\n')
                f.write(f'[### GROUNDTRUTH ###]\n{gts}\n[### /GROUNDTRUTH ###]\n')
                f.write(f'[### OUTPUT ###]\n{content_output}\n[### /OUTPUT ###]\n')
                f.write(f'[### REWARD ###]\n{reward}\n[### /REWARD ###]\n')
                f.write(f'[### EXCEPTION ###]\n{exception}\n[### /EXCEPTION ###]\n')
                f.write(f'[### /SAMPLE ###]\n')
    return rewards


def reward_segment_matching(completions, problem, solution, **kwargs):
    rewards_global_matching = reward_global_matching(completions, problem, solution, **kwargs)
    rewards_local_matching = reward_local_matching(completions, problem, solution, **kwargs)
    return [(r_g + r_l) for (r_g, r_l) in zip(rewards_global_matching, rewards_local_matching)]


reward_funcs_registry = {
    'format': reward_format,
    'format_n_timestamp': reward_format_n_timestamp,
    'segment_matching': reward_segment_matching,
}


QUESTION_TEMPLATES = {
    'frame_selection_thinking': (
        '{} First, output reasoning process in <think> </think> tags. The reasoning process must REFER TO SPECIFIC TIMESTAMPS TO TELL WHERE YOU GET THE INFORMATION FROM THE VIDEO. '
        'Then summarize your reasoning process above and output selected segments like \"<answer>X.XX-X.XX</answer>\", where \"X\" denotes arabic numbers. If there are multiple segments, separate them with spaces like \"<answer>X.XX-X.XX X.XX-X.XX</answer>\". '
        'Your output format should be like \"<think>...</think><answer>...</answer>\".'
    ),
    'frame_selection_thinking_n_timestamp': (
        '{} First, output reasoning process in <think> </think> tags. '
        'Then summarize your reasoning process above and output selected segments like \"<answer>X.XX-X.XX</answer>\", where \"X\" denotes arabic numbers. If there are multiple segments, separate them with spaces like \"<answer>X.XX-X.XX X.XX-X.XX</answer>\". '
        'Your output format should be like \"<think>...</think><answer>...</answer>\".'
    ),
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # Get question template
    question_template = QUESTION_TEMPLATES[script_args.question_template]
    dataset = DatasetDict({'train': Dataset.from_json(script_args.dataset_name)})
    mm_processor_args, video_args = ['max_pixels', 'min_pixels', 'max_frames', 'nframes', 'total_pixels'], {'type': 'video'}
    for k, v in script_args.__dict__.items():
        if k in mm_processor_args and v is not None:
            video_args[k] = v
    # Format into conversation
    def make_conversation_video(example):
        return {
            'prompt': [{
                'role': 'user',
                'content': [
                    video_args,
                    {'type': 'text', 'text': question_template.format(example['problem'])},
                ],
            }],
        }

    assert 'video' in dataset[script_args.dataset_train_split].features
    dataset = dataset.map(make_conversation_video)

    # Initialize the GRPO trainer
    trainer = GRPOVLLMTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        attn_implementation=model_args.attn_implementation,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == '__main__':
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
