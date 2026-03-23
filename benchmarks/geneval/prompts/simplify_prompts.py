import json

input_file = "/Users/gaomingju/Desktop/code/Z-Image-ours/benchmarks/geneval/prompts/evaluation_metadata_long.jsonl"
output_file = "/Users/gaomingju/Desktop/code/Z-Image-ours/benchmarks/geneval/prompts/evaluation_metadata_short.jsonl"

def generate_short_prompt(item):
    includes = item.get("include", [])
    objects = []
    for inc in includes:
        count = inc.get("count", 1)
        color = inc.get("color", "")
        cls = inc.get("class", "")
        gender = inc.get("gender", "")
        race = inc.get("race", "")
        
        desc = []
        if count > 1:
            desc.append(str(count))
        elif count == 1:
            desc.append("a")
            
        if race:
            desc.append(race)
        if gender:
            desc.append(gender)
        if color:
            desc.append(color)
        
        if cls == "person" and (gender or race):
            pass # already handled by gender/race
        else:
            if count > 1:
                if cls == "person":
                    desc.append("people")
                elif cls.endswith("s") or cls.endswith("sh") or cls.endswith("ch"):
                    desc.append(cls + "es")
                else:
                    desc.append(cls + "s")
            else:
                desc.append(cls)
                
        objects.append(" ".join(desc))
        
    if len(objects) > 1:
        obj_str = ", ".join(objects[:-1]) + " and " + objects[-1]
    else:
        obj_str = objects[0]
        
    # Fix "a Asian" -> "an Asian", etc.
    obj_str = obj_str.replace("a Asian", "an Asian")
    obj_str = obj_str.replace("a Indian", "an Indian")
    obj_str = obj_str.replace("a orange", "an orange")
    obj_str = obj_str.replace("a apple", "an apple")
    
    return f"A photo of {obj_str}."

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        item = json.loads(line)
        item["prompt"] = generate_short_prompt(item)
        f_out.write(json.dumps(item) + "\n")

print(f"Successfully generated {output_file}")
