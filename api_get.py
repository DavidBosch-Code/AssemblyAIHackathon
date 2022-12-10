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


MAX_AWAIT_SECONDS = 300  # We wait 300 seconds max
SLEEP_TIME = 5

Node = List[Dict[str, Union[str, int]]]
Links = List[Dict[str, Union[str, int]]]

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
        "auto_highlights": True
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



def process_highlights(
    model: SentenceTransformer,
    similarity_metric: Callable,
    highlights: List
) -> Tuple[Node, Links]:

    nodes = []
    links = []

    topics = [highlight["text"] for highlight in highlights]

    embedded_topics = model.encode(topics, convert_to_tensor=True)
    similarity_score = similarity_metric(
        embedded_topics, embedded_topics).numpy()

    for i, topic in enumerate(topics):

        nodes.append({
            "id": topic,
            "group": 1,
            "count": highlights[i]["count"]
        })

        for j in range(len(topics)):
            if j <= i:
                continue
            links.append({
                "source": i,
                "target": j,
                "value": highlights[i]["rank"] * highlights[j]["rank"] * similarity_score[i, j] * 100
            })

    return nodes, links
