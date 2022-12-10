from flask import Flask, request, render_template, flash, jsonify
from config import ApplicationConfig
from tempfile import TemporaryDirectory
from typing import Mapping, Any
import uuid
import json
import csv
import os
from sentence_transformers import SentenceTransformer, util
from config import CSV_CACHE, CACHE_FOLDER


import api_get

app = Flask(__name__)
app.config.from_object(ApplicationConfig)


model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_local_fn(type='mp3') -> str:
    unique_id = uuid.uuid4()
    return str(unique_id).replace('-', '') + '.' + type


def get_yt_transcription(video_url: str):

    curdir_fn = generate_local_fn()

    with TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)

        fn = os.path.join(tmpdirname, curdir_fn)

        api_get.download_yt_as_mp3(fn, video_url)

        upload_url = api_get.upload_file_to_assemblyai(fn)

        transcript_id = api_get.submit_transcription_file(upload_url)

        transcript = api_get.await_transcription(transcript_id)

    return transcript


def check_local_cache(link: str) -> Mapping[str, Any] | None:
    fn = None
    with open(CSV_CACHE, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:

            # Ignore empty rows
            if len(row) == 0:
                continue

            if row[0] == link:
                fn = row[1]
                continue
    if not fn:
        return None

    with open(fn, 'r') as f:
        transcript = json.load(f)

    return transcript


def update_local_cache(link: str, transcript: Mapping[str, Any]):
    local_fn = generate_local_fn(type='json')

    fn = os.path.join(CACHE_FOLDER, local_fn)

    with open(fn, 'w') as f:
        json.dump(transcript, f)

    with open(CSV_CACHE, 'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([link, fn])


@app.route("/", methods=["GET", "POST"])
def mainpage():

    if request.method == "POST":
        link = request.form["link"]
        if not link:
            flash("Link cannot be empty")
            return

        transcript = check_local_cache(link)
        transcript_in_cache = True if transcript else False

        if not transcript_in_cache:
            transcript = get_yt_transcription(link)
        else:
            print('loaded from cache!')

        auto_highlights = transcript['auto_highlights_result']['results']
        sentiments = transcript['sentiment_analysis_results']

        conversation, graph_data = api_get.process_highlights(
            model,
            util.cos_sim,
            auto_highlights,
            sentiments,
        )

        if not transcript_in_cache:
            update_local_cache(link, transcript)
        
        # Would make sense to cache this in the future as well
        transcript_summary = api_get.openai_summary(conversation)
        transcript_conclusions = api_get.openai_conclusions(conversation)

        return render_template(
            "homepage.html",
            link=link,
            text=conversation,
            auto_highlights=auto_highlights,
            graph_data=graph_data,
            summary=transcript_summary,
            conclusions=transcript_conclusions,
        )
    else:
        return render_template('homepage.html')


@app.route("/sendOpenAIRequest", methods = ["POST"])
def send_open_AI_request():
    question = request.json["question"]
    conversation = request.json["conversation"]

    response = api_get.openai_request(conversation + '\n\n' + question)
    response = {'value': response.choices[0].text}

    return jsonify(response), "200"


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
