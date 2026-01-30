from ollama import chat

import requests
import pandas as pd

def fix_formatting_error_Ollama(usr_prompt: str, system_prompt:str) -> str:
    model = 'gpt-oss:20b'

    messages = [
        {
            'role' : 'system',
            'content' : system_prompt
        },
        {
            'role' : 'user',
            'content': usr_prompt,
        }
    ]

    response = chat(
        model, 
        messages=messages,
        options={
        'temperature' : 1.25,
        },
    )
    return response['message']['content'] # Ollama ChatResponse does not have a reasoning field