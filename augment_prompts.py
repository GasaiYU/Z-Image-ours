import json
import random

input_file = '/Users/gaomingju/Desktop/code/Z-Image-ours/benchmarks/geneval/prompts/evaluation_metadata.jsonl'
output_file = '/Users/gaomingju/Desktop/code/Z-Image-ours/benchmarks/geneval/prompts/evaluation_metadata_long.jsonl'

suffixes = [
    ", highly detailed, 8k resolution, photorealistic, beautiful composition, cinematic lighting, shot on 35mm lens, vivid colors, masterpiece.",
    ", captured in a highly detailed professional photograph with natural lighting, sharp focus, and a slightly blurred background.",
    ", stunning visual quality, ultra-realistic, award-winning photography, perfect lighting, highly textured.",
    ", rendered in unreal engine 5, highly detailed, beautiful lighting, 4k, high quality, trending on artstation.",
    ", professional photography, studio lighting, macro lens, highly detailed, sharp focus, vibrant colors."
]

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        data = json.loads(line.strip())
        original_prompt = data['prompt']
        
        # Add a random suffix to make it longer
        suffix = random.choice(suffixes)
        data['prompt'] = original_prompt + suffix
        
        f_out.write(json.dumps(data) + '\n')

print("Done! Long prompts saved to", output_file)
