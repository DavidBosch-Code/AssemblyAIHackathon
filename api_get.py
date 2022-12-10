from typing import Mapping
from typing import Any, List, Dict, Tuple, Callable, Union

import urllib.parse
import youtube_dl
import requests
import time

from config import API_KEY
from config import ASSEMBLY_UPLOAD_ENDPOINT
from config import ASSEMBLY_TRANSCRIPT_ENDPOINT

from sentence_transformers import SentenceTransformer
import numpy as np


MAX_AWAIT_SECONDS = 300  # We wait 300 seconds max
SLEEP_TIME = 5

Node = List[Dict[str, str | int]]
Links = List[Dict[str, str | int]]


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
        'authorization': API_KEY
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
        "sentiment_analysis": True
    }

    headers = {
        "authorization": API_KEY,
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
        "authorization": API_KEY,
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


def resolve_sentiments(sentiments: List[List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
    sentiment_values = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    final_sentiments = []
    for sent_list in sentiments:
        scores = np.array([0, 0, 0], dtype=float)
        for tups in sent_list:
            if tups[0] == sentiment_values[0]:
                scores[0] += tups[1]
            elif tups[0] == sentiment_values[1]:
                scores[1] += tups[1]
            else:
                scores[2] += tups[1]
        index = np.argmax(scores)
        final_sentiments.append(
            (sentiment_values[index], scores[index]/len(sent_list)))
    return final_sentiments


def process_highlights(
    model: SentenceTransformer,
    similarity_metric: Callable,
    highlights: List,
    sentiment_analysis_results: List = None
) -> Tuple[Node, Links]:

    nodes = []
    links = []

    topics = [highlight["text"] for highlight in highlights]

    embedded_topics = model.encode(topics, convert_to_tensor=True)
    similarity_score = similarity_metric(
        embedded_topics, embedded_topics).numpy()

    if sentiment_analysis_results:
        sentiments = []
        for i, topic in enumerate(topics):
            sentiments.append([])
            for sent in sentiment_analysis_results:
                if topic in sent["text"]:
                    sentiments[i].append(
                        (sent["sentiment"], sent["confidence"]))

        sentiments = resolve_sentiments(sentiments)
        print(sentiments)

    for i, topic in enumerate(topics):

        node_dict = {
            "id": topic,
            "count": highlights[i]["count"]
        }

        if sentiment_analysis_results:
            node_dict["sentiment"] = sentiments[i][0]
            node_dict["confidence"] = sentiments[i][1]

        nodes.append(node_dict)

        for j in range(len(topics)):
            if j <= i:
                continue
            links.append({
                "source": i,
                "target": j,
                "value": highlights[i]["rank"] * highlights[j]["rank"] * similarity_score[i, j] * 100
            })

    return nodes, links
