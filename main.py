# musiCat_discord.py

import os
import time
import sqlite3
import uuid
import webbrowser
from datetime import datetime, timedelta

import discord
from discord.ext import commands
import httpx
import pandas as pd
import nltk
from nltk.corpus import words
from textblob import TextBlob
from transformers import pipeline
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# ===========================
# HARD-CODED CONFIG
# ===========================
DISCORD_TOKEN = "KEY"
OPENROUTER_API_KEY = "OPENROUTER_KEY"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

SPOTIFY_CLIENT_ID = "SPOTIFY_CLIENT_ID"
SPOTIFY_CLIENT_SECRET = "SPOTIFY_CLIENT_SECRET"
REDIRECT_URI = "https://www.google.com/"
SCOPE = "playlist-modify-public playlist-modify-private ugc-image-upload user-library-read"

PREFIX = "$"
DB_PATH = "musicat_mem.db"

MAX_HISTORY = 10
MEMORY_EXPIRE_MINUTES = 10

# ===========================
# DATABASE SETUP (SQLite)
# ===========================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute(
    """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL
)
"""
)

cur.execute(
    """
CREATE TABLE IF NOT EXISTS playlist_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    playlist_url TEXT NOT NULL,
    name TEXT,
    ts REAL NOT NULL
)
"""
)

conn.commit()


def db_add_message(user_id: str, role: str, content: str):
    now = time.time()
    cutoff = now - MEMORY_EXPIRE_MINUTES * 60
    cur.execute("DELETE FROM memory WHERE ts < ?", (cutoff,))
    conn.commit()

    cur.execute(
        "INSERT INTO memory (user_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (user_id, role, content, now),
    )
    conn.commit()


def db_get_history(user_id: str):
    now = time.time()
    cutoff = now - MEMORY_EXPIRE_MINUTES * 60
    cur.execute(
        """
        SELECT role, content FROM memory
        WHERE user_id = ? AND ts >= ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_id, cutoff, MAX_HISTORY),
    )
    rows = list(reversed(cur.fetchall()))
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def db_clear_user(user_id: str):
    cur.execute("DELETE FROM memory WHERE user_id = ?", (user_id,))
    conn.commit()


def db_log_playlist(user_id: str, playlist_url: str, name: str | None):
    cur.execute(
        "INSERT INTO playlist_log (user_id, playlist_url, name, ts) VALUES (?, ?, ?, ?)",
        (user_id, playlist_url, name, time.time()),
    )
    conn.commit()


def db_stats():
    cur.execute("SELECT COUNT(*) AS c FROM memory")
    total_messages = cur.fetchone()["c"] or 0

    cur.execute("SELECT COUNT(DISTINCT user_id) AS c FROM memory")
    total_users = cur.fetchone()["c"] or 0

    cur.execute("SELECT COUNT(*) AS c FROM playlist_log")
    total_playlists = cur.fetchone()["c"] or 0

    return total_messages, total_users, total_playlists


# ===========================
# NLP / DATA SETUP
# ===========================
nltk.download("words")
english_vocab = set(words.words())

# Load your song corpus – adjust name/path if needed
df = pd.read_csv("musicat_corpus.csv")

# Single classifier like your old API (bart-large-mnli)
classifier = pipeline("zero-shot-classification", model="M-FAC/bert-tiny-finetuned-mnli")

# Intent labels & music tags as in old API
musicat_intents = ["greeting", "conversation", "emotion", "music"]

music_tags = [
    "guitar", "classical", "slow", "techno", "strings", "drums", "electronic", "rock",
    "fast", "piano", "ambient", "beat", "violin", "vocal", "synth", "female", "indian",
    "opera", "male", "singing", "vocals", "no vocals", "harpsichord", "loud", "quiet",
    "flute", "woman", "male vocal", "no vocal", "pop", "soft", "sitar", "solo", "man",
    "classic", "choir", "voice", "new age", "dance", "male voice", "female vocal", "beats",
    "harp", "cello", "no voice", "weird", "country", "metal", "female voice", "choral",
]


def get_sentiment_score(text: str) -> float:
    return TextBlob(text).sentiment.polarity


def classify_intent(prompt: str) -> str:
    """Return top intent label: greeting / conversation / emotion / music."""
    result = classifier(prompt, musicat_intents, multi_label=True)
    tag, _ = sorted(zip(result["labels"], result["scores"]), key=lambda x: -x[1])[0]
    return tag


def is_music_intent(prompt: str) -> bool:
    tag = classify_intent(prompt)
    return tag in ["emotion", "music"]


def predict_tags(sentence: str):
    """Your old API's predict_tags logic: sentiment-aware, deprioritizing some tags."""
    sentiment = get_sentiment_score(sentence)
    result = classifier(sentence, music_tags, multi_label=True)
    preds = sorted(zip(result["labels"], result["scores"]), key=lambda x: -x[1])
    deprioritize = {"beat", "techno", "fast", "dance", "pop", "synth"}

    preds = list(preds)

    for i, (tag, score) in enumerate(preds):
        if sentiment < -0.2 and tag in deprioritize:
            preds[i] = (tag, score * 0.5)
        elif sentiment < 0.2 and tag in deprioritize:
            preds[i] = (tag, score * 0.75)

    identity_tags = {"female", "male", "man", "woman", "voice"}
    preds = [(tag, score * 0.6 if tag in identity_tags else score) for tag, score in preds]
    preds = sorted(preds, key=lambda x: -x[1])
    return [tag for tag, _ in preds[:5]]


# Your old tag compatibility map (you can extend it)
tag_compatibility = {
    ("quiet", "ambient"): 0.9,
    ("quiet", "loud"): -1.0,
    ("pop", "techno"): 0.6,
    ("metal", "ambient"): -1.0,
}


def get_compatibility(tag1: str, tag2: str) -> float:
    return (
        tag_compatibility.get((tag1, tag2))
        or tag_compatibility.get((tag2, tag1))
        or 0
    )


def get_relevant_songs(sentence: str):
    """
    This is your old API's get_relevant_songs:
    - predicted_tags
    - sentiment
    - overlap + compatibility
    - mood_penalty
    - english_vocab penalty for non-indian songs
    """
    predicted_tags = predict_tags(sentence)
    sentiment = get_sentiment_score(sentence)
    relevant = []

    for _, row in df.iterrows():
        song_tags = [t.split(" (")[0].strip() for t in row["tags"].split(";")]
        song_tag_scores = {
            t.split(" (")[0].strip(): float(t.split(" (")[1][:-1])
            for t in row["tags"].split(";")
        }

        overlap = set(predicted_tags).intersection(song_tags)
        overlap_score = sum(song_tag_scores.get(tag, 0) for tag in overlap)

        compatibility = sum(
            get_compatibility(pt, st) * song_tag_scores.get(st, 0)
            for pt in predicted_tags
            for st in song_tags
            if st not in overlap
        )

        mood_penalty = 0
        if "soft" in predicted_tags and any(
            t in song_tags for t in ["loud", "techno"]
        ):
            mood_penalty = -0.5

        total_score = (overlap_score + 0.75 * compatibility + mood_penalty) * (
            1 + sentiment
        )

        first_word = row["song_name"].split()[0].lower()
        if "indian" not in predicted_tags and first_word not in english_vocab:
            total_score *= 0.7

        if total_score > 0:
            relevant.append(
                {
                    "song_name": row["song_name"],
                    "artist_name": row["artist_name"],
                    "tags": song_tags,
                    "relevance": total_score,
                }
            )

    return sorted(relevant, key=lambda x: -x["relevance"])[:5]


# ===========================
# SPOTIFY AUTH & UTILS
# ===========================
def authenticate_spotify():
    sp_oauth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotify_token.json",
    )
    token_info = sp_oauth.get_cached_token()
    if not token_info:
        auth_url = sp_oauth.get_authorize_url()
        print("Open this URL for Spotify auth:")
        print(auth_url)
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass
        redirected = input("Paste redirected URL: ").strip()
        token_info = sp_oauth.get_access_token(redirected)

    return spotipy.Spotify(auth=token_info["access_token"])


sp = authenticate_spotify()


def search_tracks(query: str, limit: int = 10):
    results = sp.search(query, limit=limit, type="track", market="US")
    return [item["uri"] for item in results["tracks"]["items"]]


def create_playlist(song_queries: list[str], name: str | None = None):
    user_id = sp.current_user()["id"]
    if name is None:
        name = "Love from MusiCat 💘"

    playlist = sp.user_playlist_create(
        user_id,
        name,
        public=True,
        description="Made with love from MusiCat 💖",
    )

    for query in song_queries:
        uris = search_tracks(query, limit=1)
        if uris:
            sp.playlist_add_items(playlist["id"], uris)

    return playlist["external_urls"]["spotify"], name


# ===========================
# OPENROUTER (LLM)
# ===========================
async def openrouter_chat(user_id: str, text: str) -> str:
    history = db_get_history(user_id)

    messages = [
        {
            "role": "system",
            "content": (
                "You are MusiCat — a sweet, playful AI cat who loves music "
                "and chatting with cuties. Give warm, emotional, helpful replies."
            ),
        }
    ]
    messages += history
    messages.append({"role": "user", "content": text})

    async with httpx.AsyncClient(timeout=45) as client:
        try:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": OPENROUTER_MODEL, "messages": messages},
            )

            js = r.json()

            # ====== SAFETY CHECKS ======
            if "error" in js:
                print("OPENROUTER ERROR:", js["error"])
                return "Meow… I couldn’t think clearly just now. Try again? 💖"

            if "choices" not in js or len(js["choices"]) == 0:
                print("OPENROUTER EMPTY RESPONSE:", js)
                return "My brain went brrrr… can you repeat that? 😿"

            reply = js["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print("OpenRouter exception:", e)
            return "Something went meow-ssing inside my head 😿 Try again?"

    # store memory
    db_add_message(user_id, "user", text)
    db_add_message(user_id, "assistant", reply)

    return reply


async def openrouter_playlist_name(text: str) -> str:
    """Ask LLM for a short, aesthetic playlist name."""
    prompt = (
        "Based on this mood description, suggest a short, aesthetic playlist name "
        "(max 4 words, no quotes):\n\n"
        f"{text}\n\n"
        "Just answer with the name only."
    )

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a creative playlist namer."},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        js = r.json()
        name = js["choices"][0]["message"]["content"].strip()
        return name.strip('"\' ')


# ===========================
# DISCORD BOT SETUP
# ===========================
discord_intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=discord_intents, help_command=None)

class PlaylistConfirmView(discord.ui.View):
    def __init__(self, user_id: int, mood_text: str):
        super().__init__(timeout=120)
        self.user_id = user_id
        self.mood_text = mood_text

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This isn’t your button, meow 🐾", ephemeral=True
            )
            return False
        return True

    @discord.ui.button(label="Yes, make playlist 🎶", style=discord.ButtonStyle.success)
    async def yes_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)

        songs = get_relevant_songs(self.mood_text)
        if not songs:
            emb = discord.Embed(
                title="No songs found 😿",
                description="I couldn’t match anything to that mood.",
                color=discord.Color.red(),
            )
            await interaction.followup.send(embed=emb)
            self.stop()
            return

        name = await openrouter_playlist_name(self.mood_text)
        if not name:
            name = "Love from MusiCat 💘"

        song_queries = [song["song_name"] for song in songs]
        url, final_name = create_playlist(song_queries, name=name)
        db_log_playlist(str(self.user_id), url, final_name)

        emb = discord.Embed(
            title=f"🎵 {final_name}",
            description="Here’s your custom playlist:",
            color=discord.Color.green(),
        )
        emb.add_field(name="Link", value=url, inline=False)
        emb.set_footer(text="Made with love by MusiCat 💖")

        await interaction.followup.send(embed=emb)
        self.stop()

    @discord.ui.button(label="No thanks", style=discord.ButtonStyle.secondary)
    async def no_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message(
            "Okay, maybe later meow~ 🐾", ephemeral=True
        )
        self.stop()


@bot.event
async def on_ready():
    print(f"MusiCat is online as {bot.user}")


# ===========================
# COMMANDS
# ===========================
@bot.command()
async def help(ctx):
    emb = discord.Embed(
        title="MusiCat Help 😺",
        description="Commands (prefix `$`):",
        color=discord.Color.blurple(),
    )
    emb.add_field(name="$chat <message>", value="Chat with MusiCat (with memory).", inline=False)
    emb.add_field(
        name="$music <mood / description>",
        value="Get a Spotify playlist for your vibe.",
        inline=False,
    )
    emb.add_field(name="$reset", value="Reset your conversation memory.", inline=False)
    emb.add_field(name="$info", value="About MusiCat.", inline=False)
    emb.add_field(name="$stats", value="See global stats.", inline=False)
    emb.set_footer(text="Meow~")
    await ctx.reply(embed=emb)


@bot.command()
async def info(ctx):
    emb = discord.Embed(
        title="About MusiCat 🎵",
        description=(
            "MusiCat is a music-obsessed AI cat.\n"
            "It chats with you, reads your vibe, and spins playlists from your feelings."
        ),
        color=discord.Color.gold(),
    )
    emb.add_field(
        name="Memory",
        value="Remembers last 10 messages per user (for 10 minutes).",
        inline=False,
    )
    emb.add_field(
        name="Playlists",
        value="Uses your mood + tags + compatibility logic to find songs.",
        inline=False,
    )
    emb.set_footer(text="Built on your Raspberry Pi 🐾")
    await ctx.reply(embed=emb)


@bot.command()
async def stats(ctx):
    total_messages, total_users, total_playlists = db_stats()
    emb = discord.Embed(
        title="MusiCat Stats 📊",
        color=discord.Color.green(),
    )
    emb.add_field(name="Total messages", value=str(total_messages), inline=True)
    emb.add_field(name="Unique users", value=str(total_users), inline=True)
    emb.add_field(name="Playlists generated", value=str(total_playlists), inline=True)
    emb.set_footer(text="Numbers don’t lie, meow.")
    await ctx.reply(embed=emb)


@bot.command()
async def reset(ctx):
    db_clear_user(str(ctx.author.id))
    emb = discord.Embed(
        title="Memory Cleared 🧹",
        description="Forgot everything about you (for now). Start fresh, meow~",
        color=discord.Color.orange(),
    )
    await ctx.reply(embed=emb)


@bot.command()
async def chat(ctx, *, message: str):
    user_id = str(ctx.author.id)
    intent = classify_intent(message)
    sentiment = get_sentiment_score(message)

    reply = await openrouter_chat(user_id, message)

    emb = discord.Embed(
        title="MusiCat 💬",
        description=reply,
        color=discord.Color.pink(),
    )
    await ctx.reply(embed=emb)

    emotional = intent in ["emotion", "music"]
    if emotional:
        offer_emb = discord.Embed(
            title="Want a playlist for this mood? 🎶",
            description="I can spin this feeling into a Spotify playlist just for you.",
            color=discord.Color.teal(),
        )
        view = PlaylistConfirmView(ctx.author.id, message)
        await ctx.reply(embed=offer_emb, view=view)


@bot.command()
async def music(ctx, *, message: str):
    songs = get_relevant_songs(message)
    if not songs:
        emb = discord.Embed(
            title="No songs found 😿",
            description="Try describing your mood differently.",
            color=discord.Color.red(),
        )
        await ctx.reply(embed=emb)
        return

    name = await openrouter_playlist_name(message)
    if not name:
        name = "Love from MusiCat 💘"

    song_queries = [song["song_name"] for song in songs]
    url, final_name = create_playlist(song_queries, name=name)
    db_log_playlist(str(ctx.author.id), url, final_name)

    emb = discord.Embed(
        title=f"🎵 {final_name}",
        description="Here’s your custom playlist:",
        color=discord.Color.green(),
    )
    emb.add_field(name="Link", value=url, inline=False)
    emb.set_footer(text="Made with love by MusiCat 💖")
    await ctx.reply(embed=emb)


# Mention → chat + possible playlist
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    if bot.user in message.mentions and not message.content.startswith(PREFIX):
        user_id = str(message.author.id)
        content = message.content

        reply = await openrouter_chat(user_id, content)

        emb = discord.Embed(
            title="MusiCat 💬",
            description=reply,
            color=discord.Color.pink(),
        )
        await message.reply(embed=emb)

        intent = classify_intent(content)
        sentiment = get_sentiment_score(content)
        emotional = intent in ["emotion", "music"] or abs(sentiment) >= 0.4

        if emotional:
            offer_emb = discord.Embed(
                title="Playlist for this mood? 🎶",
                description="I can turn this feeling into a playlist.",
                color=discord.Color.teal(),
            )
            view = PlaylistConfirmView(message.author.id, content)
            await message.reply(embed=offer_emb, view=view)


# ===========================
# RUN BOT
# ===========================
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
