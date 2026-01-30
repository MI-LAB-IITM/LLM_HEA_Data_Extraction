import pandas as pd
from tqdm import tqdm
from modules.faissvs import FAISSVectorStore
from prompts.system import FIRST_MATCHING_PARA_PLUS_FEWSHOTS
from modules.ollama_bot import fix_formatting_error_Ollama
import sys

tqdm.pandas()

loaded_store = FAISSVectorStore.load_existing()

# Create a clean log file for messages
log_file = open('./output/llm_job.log', 'a', buffering=1)  # line-buffered

def log(msg):
    print(msg, file=log_file)
    log_file.flush()

def get_llm_response(row):
    try:
        alloys = row['Alloy'].split(" | ")
        alloys = "\n".join(alloys)
        title = row['Title']
        abstract = row['Abstract']
        extracted_section = row['Paragraph']

        few_shots = loaded_store.get_context_for_llm(row, k=2, include_extra=True)

        if abstract and extracted_section != abstract:
            prompt = f"USER INPUT\n\nENTRIES_TO_MAP\n{alloys}\n\nTITLE\n{title}\n\nABSTRACT\n{abstract}\n\nEXTRACTED TEXT SECTION:\n{extracted_section}\n\n"
        else:
            prompt = f"USER INPUT\n\nENTRIES_TO_MAP\n{alloys}\n\nTITLE\n{title}\n\nEXTRACTED TEXT SECTION:\n{extracted_section}\n\n"

        prompt = few_shots + prompt.strip()

        response = fix_formatting_error_Ollama(
            usr_prompt=prompt,
            system_prompt=FIRST_MATCHING_PARA_PLUS_FEWSHOTS
        )

        response = response.lstrip('```json').strip('```').replace('```', "").strip()
        return response
    
    except Exception as e:
        log(f"Error processing row: {e}")
        return f"ERROR: {e}"

if __name__ == "__main__":
    df_full = pd.read_excel('./output/RegexMatchingGroups.xlsx')

    # ✅ tqdm progress bar stays in terminal only
    df_full['LLM Response w Rag'] = df_full.progress_apply(get_llm_response, axis=1)


    df_full.to_excel('./output/RegexMatchingGroup_Response-1.xlsx', index=False)
    log("✅ Processing completed.")
    log_file.close()
