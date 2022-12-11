from flask import Flask, request, render_template, flash, jsonify, abort, send_file
from config import ApplicationConfig
from typing import Mapping, Any, Tuple
import uuid
import json
import csv
import os
from sentence_transformers import SentenceTransformer, util
from config import CSV_CACHE, CACHE_FOLDER
import urllib


import api_get

app = Flask(__name__)
app.config.from_object(ApplicationConfig)


model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_local_fn(type='mp3') -> str:
    unique_id = uuid.uuid4()
    return str(unique_id).replace('-', '') + '.' + type


def get_yt_transcription(video_url: str):

    curdir_fn = generate_local_fn()

    fn = os.path.join(CACHE_FOLDER, curdir_fn)

    api_get.download_yt_as_mp3(fn, video_url)

    upload_url = api_get.upload_file_to_assemblyai(fn)

    transcript_id = api_get.submit_transcription_file(upload_url)

    transcript = api_get.await_transcription(transcript_id)

    return transcript, fn


def check_local_cache(link: str) -> Tuple[Mapping[str, Any], str] | Tuple[None, None]:
    fn = None
    with open(CSV_CACHE, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:

            # Ignore empty rows
            if len(row) == 0:
                continue

            if row[0] == link:
                fn = row[1]
                mp3_loc = row[2]
                continue
    if not fn:
        return None, None

    with open(fn, 'r') as f:
        transcript = json.load(f)

    return transcript, mp3_loc


def update_local_cache(link: str, transcript: Mapping[str, Any], mp3_loc: str):
    local_fn = generate_local_fn(type='json')

    fn = os.path.join(CACHE_FOLDER, local_fn)

    with open(fn, 'w') as f:
        json.dump(transcript, f)

    with open(CSV_CACHE, 'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow([link, fn, mp3_loc])


@app.route("/", methods=["GET", "POST"])
def mainpage():

    if request.method == "POST":
        link = request.form["link"]
        if not link:
            flash("Link cannot be empty")
            return

        transcript, mp3_loc = check_local_cache(link)
        transcript_in_cache = True if transcript else False

        if not transcript_in_cache:
            transcript, mp3_loc = get_yt_transcription(link)
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
            update_local_cache(link, transcript, mp3_loc)
        
        # Would make sense to cache this in the future as well
        transcript_summary = api_get.openai_summary(conversation)
        transcript_conclusions = api_get.openai_conclusions(conversation)

        link_encoded = urllib.parse.urlencode({"link": link})

        return render_template(
            "newHomepage.html",
            link=link,
            text=conversation,
            auto_highlights=auto_highlights,
            graph_data=graph_data,
            summary=transcript_summary,
            conclusions=transcript_conclusions,
            mp3_loc=link_encoded,
        )
    else:
        return render_template('newHomepage.html')


@app.route('/return_mp3')
def return_mp3():
    link = request.args.get("link")
    _, mp3_loc = check_local_cache(link)
    if not mp3_loc:
        abort(404)
    try:
        return send_file(mp3_loc)
    except Exception as e:
        return str(e)


@app.route("/sendOpenAIRequest", methods = ["POST"])
def send_open_AI_request():
    question = request.json["question"]
    conversation = request.json["conversation"]

    response = api_get.openai_request(conversation + '\n\n' + question)
    response = {'value': response.choices[0].text}

    return jsonify(response), "200"


@app.route("/sendUpdatedTranscript", methods = ["POST"])
def reevaluate_transcript():
    updated_transcript = request.json["transcript"]
    updated_transcript = updated_transcript.split("\n")
    updated_transcript = [sentence.strip() for sentence in updated_transcript if sentence.strip()]

    transcript_summary = api_get.openai_summary(updated_transcript)
    transcript_conclusions = api_get.openai_conclusions(updated_transcript)

    response = {
        'summary': transcript_summary,
        'conclusions': transcript_conclusions
    }

    return jsonify(response), "200"


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
