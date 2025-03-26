import os
from pathlib import Path
from dotenv import load_dotenv
import markdown
from openai import OpenAI
import torch
from TTS.api import TTS

# === ENVIRONMENT SETUP ===

def initialize_openai_client():
    """Loads API key from .env and returns a configured OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def initialize_tts_model():
    """Load the multilingual XTTS-v2 TTS model with GPU/CPU fallback."""
    try:
        return TTS(model_name="tts_models/multilingual/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Warning: Could not load XTTS model. {e}")
        return None


# === FILE UTILITIES ===

def load_file(path):
    """Reads a text file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# === GPT PROCESSING ===

def build_prompt(prompt_template, topic, supplementary_info):
    """Fills placeholders in a template with the given topic and info."""
    return prompt_template.format(topic=topic, supplementary_info=supplementary_info)


def send_prompt_to_openai(prompt, model="gpt-4o"):
    """Sends a prompt to OpenAI and returns GPT's response."""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


# === HTML UTILITIES ===

def convert_markdown_to_html(markdown_text):
    """Converts Markdown text to HTML."""
    return markdown.markdown(markdown_text)


def inject_content_into_html(html_template, content_html):
    """Replaces {{CONTENT}} in HTML template with GPT response HTML."""
    return html_template.replace("{{CONTENT}}", content_html)


# === TTS UTILITIES ===

def extract_first_paragraph(markdown_text):
    """Returns the first non-empty paragraph from Markdown."""
    paragraphs = [p.strip() for p in markdown_text.split('\n\n') if p.strip()]
    return paragraphs[0] if paragraphs else ""


def synthesize_speech(text, speaker_wav, output_path, language="nl"):
    """Generates a spoken audio file using XTTS and saves it as WAV."""
    if not tts_model:
        print("TTS model not loaded; skipping audio synthesis.")
        return
    tts_model.tts_to_file(
        text=text,
        speaker_wav=str(speaker_wav),
        language=language,
        file_path=str(output_path)
    )


# === MAIN ORCHESTRATION ===

def run_trend_query(topic, prompt_template_path, supplementary_info_path,
                    html_template_path, speaker_wav_path, language_code="nl"):
    """
    Generates GPT response and TTS narration for a given topic.
    Saves HTML and WAV files locally.
    """
    output_stem = f"output_{topic.replace(' ', '_')}"
    output_html_file = f"{output_stem}.html"
    output_wav_file = f"{output_stem}.wav"

    try:
        # Load required template and input files
        prompt_template = load_file(prompt_template_path)
        supplementary_info = load_file(supplementary_info_path)
        html_template = load_file(html_template_path)

        # Generate prompt and send to GPT
        prompt = build_prompt(prompt_template, topic, supplementary_info)
        gpt_markdown_response = send_prompt_to_openai(prompt)
        print("=== GPT Response (Markdown) ===\n", gpt_markdown_response)

        # Convert and inject HTML
        converted_html = convert_markdown_to_html(gpt_markdown_response)
        final_html = inject_content_into_html(html_template, converted_html)

        with open(output_html_file, "w", encoding="utf-8") as f:
            f.write(final_html)
        print(f"✅ HTML saved to: {output_html_file}")

        # Extract first paragraph and synthesize speech
        first_paragraph = extract_first_paragraph(gpt_markdown_response)
        if first_paragraph:
            print("=== First Paragraph ===\n", first_paragraph)
            synthesize_speech(
                text=first_paragraph,
                speaker_wav=speaker_wav_path,
                output_path=output_wav_file,
                language=language_code
            )
            print(f"✅ Speech saved to: {output_wav_file}")
        else:
            print("⚠ No paragraph found for TTS.")

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


# === SCRIPT ENTRY POINT ===

if __name__ == "__main__":
    # Initialize clients
    client = initialize_openai_client()
    tts_model = initialize_tts_model()

    # Input parameters
    topic = "algorithmic bias"
    prompt_template_path = Path("../docs/prompt_template.txt")
    supplementary_info_path = Path("docs/org_structure_amsterdam.txt")
    html_template_path = Path("../docs/template.html")
    speaker_wav_path = Path("../audio/Chriet_Titulaer.wav")
    language_code = "nl"

    run_trend_query(
        topic,
        prompt_template_path,
        supplementary_info_path,
        html_template_path,
        speaker_wav_path,
        language_code
    )
