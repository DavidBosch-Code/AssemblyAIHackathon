from typing import Mapping
from typing import Any, List, Dict, Tuple, Callable

import numpy as np
import urllib.parse
import youtube_dl
import requests
import time

from config import OPENAI_API_KEY
from config import ASSEMBLY_API_KEY
from config import ASSEMBLY_UPLOAD_ENDPOINT
from config import ASSEMBLY_TRANSCRIPT_ENDPOINT

from sentence_transformers import SentenceTransformer
import numpy as np
from openai.openai_response import OpenAIResponse
import openai
import math


openai.api_key = OPENAI_API_KEY

MAX_AWAIT_SECONDS = 300  # We wait 300 seconds max
SLEEP_TIME = 5
OPENAI_MODEL = 'text-curie-001'
MAX_TOKEN_COUNT = 2000
MAX_DEPTH = 4


def download_yt_as_mp3(out_fn: str, video_url: str):

    video_info = youtube_dl.YoutubeDL().extract_info(
        url=video_url, download=False
    )

    options = {
        'format': 'bestaudio/best',
        'keepvideo': False,
        'outtmpl': out_fn,
        'rm-cache-dir': True,
    }

    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])


def upload_file_to_assemblyai(fn: str, chunk_size: int = 5242880) -> str:
    def read_file_chunks():
        with open(fn, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data

    headers = {
        'authorization': ASSEMBLY_API_KEY
    }
    response = requests.post(ASSEMBLY_UPLOAD_ENDPOINT,
                             headers=headers,
                             data=read_file_chunks())

    print(response.json())
    upload_url = response.json()['upload_url']

    return upload_url


def submit_transcription_file(upload_url: str) -> str:
    # TODO: Add option to get sentiment analysis & key words/phrases
    json = {
        "audio_url": upload_url,
        "auto_highlights": True,
        "sentiment_analysis": True,
        "speaker_labels": True
    }

    headers = {
        "authorization": ASSEMBLY_API_KEY,
        "content-type": "application/json"
    }

    response = requests.post(
        ASSEMBLY_TRANSCRIPT_ENDPOINT, json=json, headers=headers)

    print(response)
    print(response.json())
    transcript_id = response.json()['id']

    return transcript_id


def await_transcription(transcript_id: str) -> Mapping[str, Any]:

    endpoint = urllib.parse.urljoin(
        ASSEMBLY_TRANSCRIPT_ENDPOINT + '/', transcript_id)
    print(endpoint)

    headers = {
        "authorization": ASSEMBLY_API_KEY,
    }

    start_time = time.time()
    completed_result = False

    while time.time() - start_time < MAX_AWAIT_SECONDS:
        response = requests.get(endpoint, headers=headers)

        result = response.json()['status']

        # Temporary logging
        print('Result:', result)

        if result == 'completed':
            completed_result = True
            break

        time.sleep(SLEEP_TIME)

    if not completed_result:
        raise Exception('Took too long to process file!')

    return response.json()


def resolve_sentiments(sentiments_dict: Dict[str, List[List[Tuple[str, float]]]]) -> Dict[str, List[Tuple[str, float]]]:
    return_dict = {}
    for key, value in sentiments_dict.items():
        return_dict[key] = _resolve_sentiments(value)
    return return_dict


def _resolve_sentiments(sentiments: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    sentiment_values = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    final_sentiments = []
    for sent_list in sentiments:
        if not sent_list:
            final_sentiments.append(("NA", 0))
            continue
        scores = np.array([0, 0, 0], dtype=float)
        for tups in sent_list:
            if tups[0] == sentiment_values[0]:
                scores[0] += tups[1]
            elif tups[0] == sentiment_values[1]:
                scores[1] += tups[1]
            elif tups[0] == sentiment_values[2]:
                scores[2] += tups[1]
        index = np.argmax(scores)
        final_sentiments.append(
            (sentiment_values[index], scores[index]/len(sent_list)))
    return final_sentiments


def process_highlights(
    model: SentenceTransformer,
    similarity_metric: Callable,
    highlights: List[Mapping[str, Any]],
    sentiments: List[Mapping[str, Any]],
    threshold: float = 0.25
) -> Tuple[List[str], Dict]:

    nodes = {}
    links = {}
    sentiments_list = {}

    conversation = []
    topics = [highlight["text"] for highlight in highlights]

    for sentiment in sentiments:
        speaker = sentiment["speaker"]
        conversation.append("Speaker " + speaker + ": " + sentiment["text"])
        if speaker not in nodes:
            nodes[speaker] = []
            links[speaker] = []
            sentiments_list[speaker] = [[] for _ in range(len(topics))]

        for i, topic in enumerate(topics):
            if topic in sentiment["text"]:
                sentiments_list[speaker][i].append(
                    (sentiment["sentiment"], sentiment["confidence"]))

    sentiments_list = resolve_sentiments(sentiments_list)
    embedded_topics = model.encode(topics, convert_to_tensor=True)
    similarity_score = similarity_metric(
        embedded_topics, embedded_topics).numpy()

    counter = {}
    for speaker in nodes:
        counter[speaker] = 0

    for i, topic in enumerate(topics):
        for speaker in nodes:
            if sentiments_list[speaker][i][0] == "NA":
                counter[speaker] += 1
                continue

            node_dict = {
                "id": topic,
                "count": highlights[i]["count"],
                "sentiment": sentiments_list[speaker][i][0],
                "confidence": sentiments_list[speaker][i][1]
            }

            nodes[speaker].append(node_dict)
            second_counter = counter[speaker]
            for j in range(len(topics)):
                if j <= i:
                    continue
                if sentiments_list[speaker][j][0] == "NA":
                    second_counter += 1
                    continue
                value = similarity_score[i,
                                         j] if similarity_score[i, j] >= threshold else 0
                links[speaker].append({
                    "source": i - counter[speaker],
                    "target": j - second_counter,
                    "value": highlights[i]["rank"] * highlights[j]["rank"] * value * 100
                })

    return_dict = {}
    for speaker in nodes:
        return_dict[speaker] = {
            "nodes": nodes[speaker],
            "links": links[speaker]
        }

    return conversation, return_dict


def get_conversation_word_count(conversation: List[str]) -> str:
    word_count = 0
    for sentence in conversation:
        word_count += len(sentence.split())
    return word_count


def openai_condense(conversation: List[str], depth=0) -> str:

    if depth >= MAX_DEPTH:
        raise Exception("Stop feeding me such long videos!")

    word_count = get_conversation_word_count(conversation)

    max_word_count = MAX_TOKEN_COUNT // 2

    num_sections = math.ceil(word_count / max_word_count)

    if num_sections == 1:
        return conversation

    words_per_section = word_count // num_sections

    print(num_sections, words_per_section)

    i = 0
    cur_section = []
    num_words_in_cur_section = 0
    final_result = []

    task = "Rewrite the following conversation, removing uninteresting and irrelevant sentences:\n\n"

    while i < len(conversation):
        cur_section.append(conversation[i])
        num_words_in_cur_section += len(conversation[i].split())
        if num_words_in_cur_section >= words_per_section:
            prompt = task + '\n'.join(cur_section)
            response = openai_request(
                prompt, max_tokens=512, temperature=0.2).choices[0].text
            final_result.extend(response.split("\n"))
            num_words_in_cur_section = 0
            cur_section = []
        i += 1

    print('\n'.join(final_result))

    return openai_condense(final_result, depth=depth+1)


def openai_summary(conversation: List[str]) -> str:
    task = "CONVERSATION START\n"
    prompt = task + \
        '\n'.join([val for val in conversation if val.strip()]) + \
        "\nCONVERSATION END"
    prompt += "\n\nSummarize the prior conversation into a few sentences.\n"

    response = openai_request(prompt)

    return response.choices[0].text


def openai_conclusions(conversation: List[str]) -> str:
    task = "CONVERSATION START\n"
    prompt = task + \
        '\n'.join([val for val in conversation if val.strip()]) + \
        "\nCONVERSATION END"
    prompt += "\n\List the top three conclusions a reader can take away from the following conversation.\n"

    response = openai_request(prompt)

    return response.choices[0].text


def openai_request(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> OpenAIResponse:
    print(prompt)

    response = openai.Completion.create(
        model=OPENAI_MODEL,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    print(response)

    return response
