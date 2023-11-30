import openai
import json

openai.api_key = "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"


def get_gpt_generation(prompt,
                modelname='gpt-4'):
    res = openai.chat.completions.create(
        model=modelname,
        messages=[
                {"role": "system", "content":  "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    json_res = json.loads(res.model_dump_json())
    generation = json_res["choices"][0]["message"]["content"]

    return generation
