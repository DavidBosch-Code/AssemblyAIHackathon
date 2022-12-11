# AssemblyAIHackathon: Interview Analysis & Exploration Dashboard

This project provides a dashboard to analyze interviews. Given a YouTube video link with two speakers, it will provide a transcription (using AssemblyAI) and graphs of the key phrases that each speaker mentioned. We also provide a summary and key takeaways using OpenAI. There is an option to update the transcript, in case there are any issues, that will update the summary and takeaways. The user also has the option to ask further questions about the interview, which are answered through OpenAI.

The graph links are generated based on the cosine similarity of the phrase embeddings (using a simple sentence transformers model).

![image](https://user-images.githubusercontent.com/49696908/206913116-bc416ddc-37bd-4aa2-bbe9-82ad2d2b4050.png)

# Motivation

In many domains, especially in the social sciences, analysing and transcribing interviews is a time consuming and costly task that often requires mutliple tools (e.g., separate transcription & analysis software, with much manual effort). This dashboard provides a very simple MVP/POC that could help in such cases. The interactive QA and summarization with large language models can aid analysis, especially for longer interviews.

# Setup

1. Install all requirements (ideally in a virtual env)
    pip install -r requirements.txt
2. Create a new file .env and add your Assembly AI and OpenAI API keys to it
    ASSEMBLY_API_KEY=...
    OPENAI_API_KEY=...
3. Run server.py

Now you can begin using the dashboard! Just provide a full YouTube link and it should start working.

# Notes

- Downloading from YouTube can be quite slow, especially for longer videos. You can see the progress in the command prompt/terminal.
- We save the video mp3s & transcript results, so running the dashboard on a previously used link should be faster (it does re-run OpenAI requests)
- We assume each video has exactly two speakers. More or less may lead to unexpected behavior
