# from transformers import pipeline
# import json

# import os

# pipe = pipeline(
#     task="automatic-speech-recognition",
#     model="openai/whisper-base"
# )

# audios=os.listdir("audios")
# for audio in audios:
#     number=audio.split("_")[0]
#     title=audio.split("_")[1][:-4]   
#     print(number,title) 
#     result = pipe(f"audios/{audio}", generate_kwargs={"task": "translate", "language": "hi"},return_timestamps=True)
#     with open(f"jsons/{number}_{title}.json","w") as f:
#         json.dump(result,f)

from transformers import pipeline
import json
import os
import re


pipe = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-base"
)


os.makedirs("jsons", exist_ok=True)


audios = os.listdir("audios")

for audio in audios:
    number = audio.split("_")[0]
    title = os.path.splitext("_".join(audio.split("_")[1:]))[0]  


    safe_title = re.sub(r'[\\/*?:"<>|,& ]+', "_", title)

    print(f" Processing: {number} - {title}")

    try:

        result = pipe(
            f"audios/{audio}",
            generate_kwargs={"task": "translate", "language": "hi"},
            return_timestamps=True
        )

        
        with open(f"jsons/{number}_{safe_title}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"✅ Saved: jsons/{number}_{safe_title}.json")

    except Exception as e:
        print(f"❌ Error processing {audio}: {e}")
