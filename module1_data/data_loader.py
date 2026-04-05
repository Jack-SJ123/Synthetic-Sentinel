"""
data_loader.py — Synthetic Sentinel Data Pipeline
===================================================
Phase 1: Data acquisition, cleaning, tokenization, feature engineering, and splitting.

Handles three data sources:
  1. TweepFake (HuggingFace `datasets`)
  2. FakeNewsNet-style news articles (bundled CSV / synthetic fallback)
  3. Custom AI-generated samples (generated locally via simple templates)

Outputs:
  data/train.csv, data/val.csv, data/test.csv
  Each with columns: text, label (0=human, 1=AI), source, perplexity, burstiness
"""

import os
import json
import re
import math
import random
import warnings
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    RobertaTokenizerFast,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MAX_TOKEN_LEN = 512
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1 · Dataset acquisition
# ---------------------------------------------------------------------------

def load_tweepfake() -> pd.DataFrame:
    """
    Load the TweepFake dataset from HuggingFace `datasets` library.
    Falls back to a synthetic stand-in if the dataset is unavailable.
    """
    print("[1/6] Loading TweepFake dataset ...")
    try:
        from datasets import load_dataset
        ds = load_dataset("twturbulence/TweepFake", split="train", trust_remote_code=True)
        df = ds.to_pandas()
        # Standardise columns
        if "text" not in df.columns:
            text_col = [c for c in df.columns if c.lower() in ("tweet", "content", "text")]
            if text_col:
                df.rename(columns={text_col[0]: "text"}, inplace=True)
        if "label" not in df.columns:
            label_col = [c for c in df.columns if "label" in c.lower() or "account" in c.lower()]
            if label_col:
                df.rename(columns={label_col[0]: "label"}, inplace=True)
        # Convert labels: 'bot' / 1 -> 1, 'human' / 0 -> 0
        if df["label"].dtype == object:
            df["label"] = df["label"].map(lambda x: 1 if str(x).lower() in ("bot", "1", "ai", "machine") else 0)
        df["source"] = "tweepfake"
        print(f"   -> loaded {len(df)} samples from TweepFake")
        return df[["text", "label", "source"]]
    except Exception as e:
        print(f"   [!] TweepFake unavailable ({e}); generating synthetic tweets ...")
        return _generate_synthetic_tweets(5000)


def _generate_synthetic_tweets(n: int) -> pd.DataFrame:
    """Generate simple synthetic tweet-like data as a fallback using permutations."""
    templates = [
        ("human", "Just had the best {food} at {place}! ☕ #{adj}vibes"),
        ("human", "Can't believe {team} lost again 😤 what a {adj} disaster"),
        ("human", "Anyone else think {topic} is getting {adj}? Just me? Ok."),
        ("human", "Lmao my {relative} just asked me what a {thing} is 😂"),
        ("human", "Hot take: {food} at {place} is {adj}. Fight me."),
        ("human", "Woke up at {time} and chose chaos today 🔥 {topic} is wild"),
        ("human", "Thinking about quitting my job to become a {career} {time} honestly"),
        ("human", "Why does {thing} always happen on Mondays smh #relatable"),
        ("human", "This weather is absolutely {adj}! Perfect for {activity} with {relative}."),
        ("human", "So {person} really said that huh... {adj} times we live in"),
        ("ai", "The importance of {topic} cannot be overstated in today's society. It is {adj}."),
        ("ai", "A comprehensive analysis of {topic} reveals several {adj} factors at {place}."),
        ("ai", "It is worth noting that {topic} has shown {adj} potential for growth."),
        ("ai", "In conclusion, {topic} represents a {adj} opportunity for {person}."),
        ("ai", "The relationship between {topic} and {thing} is well-documented in {place}."),
        ("ai", "One must consider the {adj} nature of {topic} when evaluating its impact."),
        ("ai", "Research by {person} suggests that {topic} plays a {adj} role in shaping outcomes."),
        ("ai", "The evolution of {topic} has been marked by {adj} milestones since {time}."),
        ("ai", "Understanding {topic} requires a {adj} approach involving {career}s."),
        ("ai", "Recent developments in {topic} highlight the need for {adj} frameworks at {place}."),
    ]
    fillers = {
        "place": ["Starbucks", "downtown", "school", "work", "the office", "the city", "London", "NY", "home", "campus", "the lab", "the clinic"],
        "team": ["the Lakers", "my fantasy team", "United", "the Leafs", "the Eagles", "the Bulls", "City", "Arsenal"],
        "topic": ["AI regulation", "climate change", "remote work", "social media", "healthcare", "the economy", "education", "space travel"],
        "relative": ["mom", "dad", "grandma", "uncle", "sister", "brother", "cousin", "roommate"],
        "food": ["sushi", "pizza", "avocado toast", "smoothies", "tacos", "pasta", "burgers", "ramen"],
        "time": ["4am", "5:30am", "noon", "3am", "midnight", "6am", "dusk", "dawn"],
        "career": ["barista", "dog walker", "creator", "farmer", "dev", "artist", "writer", "chef"],
        "thing": ["this", "bad luck", "traffic", "drama", "inflation", "lag", "the bug", "the email"],
        "adj": ["gorgeous", "terrible", "wild", "perfect", "crucial", "significant", "complex", "unprecedented", "massive"],
        "activity": ["hiking", "reading", "napping", "coding", "gaming", "running", "cooking", "sleeping"],
        "person": ["they", "the CEO", "that influencer", "my boss", "the mayor", "the director", "experts", "scientists"],
    }

    import itertools
    rows = []
    generated = set()
    
    while len(rows) < n:
        label_type, template = random.choice(templates)
        
        # Fill template
        text = template
        for match in set(re.findall(r"\{(\w+)\}", text)):
            if match in fillers:
                text = text.replace("{" + match + "}", random.choice(fillers[match]), 1)
                
        if text not in generated:
            generated.add(text)
            label = 0 if label_type == "human" else 1
            rows.append({"text": text, "label": label, "source": "tweepfake_synthetic"})
            
    return pd.DataFrame(rows)


def load_fake_news() -> pd.DataFrame:
    """
    Load news-article-level data.  Uses a synthetic generation if no local CSV exists.
    """
    print("[2/6] Loading news article dataset ...")
    local_csv = os.path.join(DATA_DIR, "fakenews_raw.csv")
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv)
        df["source"] = "fakenewsnet"
        print(f"   -> loaded {len(df)} articles from local CSV")
        return df[["text", "label", "source"]]

    # Fallback: generate synthetic news snippets
    print("   [!] No local news CSV; generating synthetic news articles ...")
    return _generate_synthetic_news(3000)


def _generate_synthetic_news(n: int) -> pd.DataFrame:
    """Generate realistic-length synthetic news snippets using permutations."""
    human_templates = [
        "After months of debate in {city}, the council voted to approve a new {project}. The plan includes {detail}. Opponents argued the cost was too high, while supporters pointed to {benefit}. 'This is a win for {group},' said {official}.",
        "{business}, a local {industry} business, received the {award} this week. Owner {person} has focused on {focus} since opening in {year}. 'We prove that {concept} can work,' they told reporters in {city}.",
        "{company}, a startup founded by {group}, raised {amount} to improve {project}. The company's technology reduces costs by {percent}%. Initial deployments are planned for {city} and surrounding areas.",
        "A devastating {disaster} swept through {city} today, causing {amount} in damages. Emergency services responded quickly, rescuing {group} from the affected {project}. Mayor {official} declared a state of emergency.",
        "The community of {city} came together to celebrate the annual {event}. Thousands of {group} attended, enjoying {detail} and {benefit}. The highlight was a performance by {person}, which drew record crowds.",
    ]
    ai_templates = [
        "Global markets demonstrated notable resilience regarding {industry}, with indices recording gains. Analysts attribute this to {benefit} and {detail}. The {project} sector showed robust performance, driven by {concept}.",
        "A comprehensive study on {project} found that it can lead to increases in {benefit} when implemented properly. The study surveyed {group} and identified {concept} as a key factor in {city}.",
        "Authorities have issued updated guidelines for {industry} that emphasize {concept}. The recommendations reflect a combination of {detail} and {benefit}. Experts note this addresses concerns raised by {group}.",
        "Recent advancements in {industry} highlight the potential for {concept}. The integration of {project} offers substantial {benefit}. Stakeholders across {city} are monitoring these developments closely.",
        "The intersection of {project} and {industry} presents unique challenges for {group}. Addressing these requires a focus on {concept} and {detail}. Researchers estimate a {percent}% improvement is possible.",
    ]
    
    fillers = {
        "city": ["Springfield", "Shelbyville", "Capital City", "Oakhaven", "Riverdale", "Metropolis", "Gotham", "Star City"],
        "project": ["transit expansion", "internet access", "infrastructure repair", "housing initiative", "education reform", "green energy grid", "healthcare facility"],
        "detail": ["extended evening service", "mesh network integration", "tax incentives", "community workshops", "AI-driven analytics", "subsidized programs"],
        "benefit": ["increased ridership", "digital inclusion", "economic growth", "reduced emissions", "improved public health", "greater efficiency"],
        "group": ["working families", "rural communities", "recent graduates", "local residents", "small business owners", "policymakers", "industry leaders"],
        "official": ["Councilor Martinez", "Mayor Quimby", "Director Vance", "Senator Davis", "Governor Thorne"],
        "business": ["Green Fork Bistro", "ConnectAll Networks", "Apex Solutions", "Pioneer Tech", "Summit Industries"],
        "industry": ["hospitality", "telecommunications", "renewable energy", "biotech", "manufacturing", "finance"],
        "award": ["sustainability award", "innovation prize", "community excellence medal", "startup of the year", "top employer accolade"],
        "person": ["Jamie Chen", "Dr. Aris", "Sarah Jenkins", "Michael Chang", "Prof. Rivera"],
        "focus": ["local sourcing", "affordable pricing", "zero-waste operations", "inclusive hiring", "open-source development"],
        "year": ["2019", "2021", "2023", "2015", "last year"],
        "concept": ["responsibility and profit", "connectivity for all", "sustainable growth", "equitable access", "technological synergy"],
        "amount": ["$2 million", "$500,000", "$10 million", "unprecedented funding", "significant capital"],
        "percent": ["60", "25", "40", "15", "80"],
        "disaster": ["flood", "storm", "wildfire", "power outage"],
        "event": ["summer festival", "tech expo", "cultural parade", "marathon"],
    }

    rows = []
    generated = set()
    
    while len(rows) < n:
        label = random.choice([0, 1])
        template = random.choice(human_templates if label == 0 else ai_templates)
        
        text = template
        for match in set(re.findall(r"\{(\w+)\}", text)):
            if match in fillers:
                text = text.replace("{" + match + "}", random.choice(fillers[match]), 1)
                
        if text not in generated:
            generated.add(text)
            rows.append({"text": text, "label": label, "source": "fakenews_synthetic"})
            
    return pd.DataFrame(rows)


def generate_custom_synthetic(n: int = 1000) -> pd.DataFrame:
    """
    Generate custom AI-like vs human-like samples using template variation.
    """
    print(f"[3/6] Generating {n} custom synthetic samples ...")
    human_templates = [
        "I've been thinking about {topic}, and honestly I don't think there's one answer. My {relative} was a {adj} person—never raised their voice—but everyone respected them. Maybe it's just about {concept}? Idk.",
        "So I tried {activity} this weekend and it was a DISASTER. The {thing} was {adj}, and somehow I messed up the {detail}. But hey, at least the {benefit} turned out fine. Will I try again? Probably not lol.",
        "The {event} last night was absolutely unreal. Like, the whole place turned this crazy shade of {adj} and I just stood there for 20 minutes. Sometimes you forget how beautiful {city} is. Needed that moment.",
        "Can we talk about how {adj} the new {project} is? I waited in line for {amount} just to see it. Totally worth it though, the {detail} was incredible. Highly recommend if you're in {city}.",
        "My {relative} literally just texted me asking if {topic} is real. I can't even right now 😂. How do I explain {concept} to someone who still uses a flip phone?",
    ]
    ai_templates = [
        "{topic} is a multifaceted concept that encompasses {concept} and {detail}. Effective implementation demonstrates {benefit} in the face of changing circumstances. Research shows it achieves superior outcomes in {city}.",
        "{activity} has gained significant popularity as a strategy for {benefit}. The process involves {concept} and optimizing time management. Studies indicate that individuals who engage in this report higher satisfaction.",
        "The phenomenon of {event} is a well-documented effect that occurs when {detail} interacts with {industry}. The {adj} nature of this process produces the displays commonly observed in {city}.",
        "Analyzing the {project} reveals critical insights into {topic}. The integration of {concept} provides a {adj} framework for understanding {benefit}. Stakeholders must consider these variables carefully.",
        "Evaluating {topic} requires recognizing the paramount importance of {concept}. Systematic application of {detail} yields a {percent}% improvement in {benefit}, underscoring the necessity of robust methodologies.",
    ]
    
    fillers = {
        "topic": ["leadership", "meal prepping", "sunset coloration", "productivity", "AI ethics", "mindfulness", "urban planning"],
        "relative": ["grandpa", "aunt", "cousin", "manager", "neighbor"],
        "adj": ["quiet", "crunchy", "vibrant", "revolutionary", "perplexing", "stunning", "essential", "dynamic"],
        "concept": ["showing up", "following instructions", "Rayleigh scattering", "transparent communication", "strategic vision", "dietary tracking"],
        "activity": ["baking bread", "learning Python", "gardening", "meditation retreat", "yoga"],
        "thing": ["dough", "compiler", "soil", "cushion", "mat"],
        "detail": ["kneading process", "syntax error", "water level", "breathing technique", "posture"],
        "benefit": ["crust", "logic", "flowers", "relaxation", "flexibility"],
        "event": ["meteor shower", "product launch", "concert", "art exhibition"],
        "city": ["Seattle", "Austin", "Chicago", "Boston", "Denver"],
        "project": ["software update", "museum exhibit", "park renovation", "transit line"],
        "amount": ["2 hours", "45 minutes", "all day", "three weeks"],
        "industry": ["astronomy", "technology", "entertainment", "hospitality"],
        "percent": ["20", "35", "50", "75"],
    }

    rows = []
    generated = set()
    
    while len(rows) < n:
        label = random.choice([0, 1])
        template = random.choice(human_templates if label == 0 else ai_templates)
        
        text = template
        for match in set(re.findall(r"\{(\w+)\}", text)):
            if match in fillers:
                text = text.replace("{" + match + "}", random.choice(fillers[match]), 1)
                
        if text not in generated:
            generated.add(text)
            rows.append({"text": text, "label": label, "source": "custom_synthetic"})
            
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2 · Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """De-noise a single text sample."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)            # HTML tags
    text = re.sub(r"https?://\S+", "", text)        # URLs
    text = re.sub(r"@\w+", "", text)                # @handles
    text = re.sub(r"#\w+", "", text)                # hashtags (keep for tweets? optional)
    text = re.sub(r"\s+", " ", text).strip()         # collapse whitespace
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning, drop empties and duplicates."""
    print("[4/6] Cleaning and de-duplicating ...")
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 20].drop_duplicates(subset="text").reset_index(drop=True)
    print(f"   -> {len(df)} samples after cleaning")
    return df


def build_dataset_metadata(df: pd.DataFrame) -> dict:
    """Summarize dataset composition and whether synthetic fallback data was used."""
    source_counts = {str(k): int(v) for k, v in df["source"].value_counts().to_dict().items()}
    label_counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
    synthetic_sources = [source for source in source_counts if "synthetic" in source]
    real_sources = [source for source in source_counts if "synthetic" not in source]

    return {
        "total_samples": int(len(df)),
        "source_counts": source_counts,
        "label_counts": label_counts,
        "synthetic_sources": synthetic_sources,
        "real_sources": real_sources,
        "synthetic_fraction": float(
            sum(source_counts[source] for source in synthetic_sources) / max(len(df), 1)
        ),
    }


# ---------------------------------------------------------------------------
# 3 · Feature engineering
# ---------------------------------------------------------------------------

def compute_burstiness(text: str) -> float:
    """
    Burstiness = std-dev of sentence lengths (word count per sentence).
    Human text tends to have higher burstiness (varied sentence structure).
    """
    sentences = re.split(r"[.!?]+", text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths))


def compute_perplexity_batch(texts: list[str], model, tokenizer, device: str, batch_size: int = 8) -> list[float]:
    """
    Compute pseudo-perplexity for each text using a small GPT-2 model.
    AI-generated text typically has LOWER perplexity (more predictable).
    """
    perplexities = []
    model.eval()
    for i in tqdm(range(0, len(texts), batch_size), desc="   Computing perplexity", leave=False):
        batch = texts[i : i + batch_size]
        encodings = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            # Per-sample loss isn't directly available in batched mode,
            # so we compute one sample at a time for accuracy.
        for text in batch:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model(**enc, labels=enc["input_ids"])
            ppl = math.exp(min(out.loss.item(), 100))  # cap to avoid overflow
            perplexities.append(ppl)
    return perplexities


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add perplexity and burstiness columns."""
    print("[5/6] Computing features (perplexity + burstiness) ...")

    # Burstiness — fast, CPU-only
    df["burstiness"] = df["text"].apply(compute_burstiness)

    # Perplexity — requires GPT-2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    df["perplexity"] = compute_perplexity_batch(df["text"].tolist(), model, tokenizer, device)

    # Free GPU memory
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"   -> Features added: perplexity (mean={df['perplexity'].mean():.1f}), "
          f"burstiness (mean={df['burstiness'].mean():.2f})")
    return df


# ---------------------------------------------------------------------------
# 4 · Train / Val / Test split
# ---------------------------------------------------------------------------

def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    """70/15/15 stratified split -> CSV."""
    print("[6/6] Splitting data (70/15/15) and saving ...")
    os.makedirs(output_dir, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=RANDOM_SEED
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"   -> train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    print(f"   -> Saved to {output_dir}/")

    # Label distribution
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df["label"].value_counts().to_dict()
        print(f"   -> {name}: human={counts.get(0, 0)}, AI={counts.get(1, 0)}")

    metadata = build_dataset_metadata(df)
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"   -> Saved dataset metadata: {metadata_path}")

    if metadata["synthetic_fraction"] > 0:
        print(
            "   [!] Synthetic fallback data detected. Report results as template/synthetic-benchmark "
            "performance, not as broad real-world generalization."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic Sentinel — Data Pipeline")
    parser.add_argument("--skip-perplexity", action="store_true",
                        help="Skip perplexity computation (faster, for testing)")
    parser.add_argument("--custom-n", type=int, default=1000,
                        help="Number of custom synthetic samples to generate")
    args = parser.parse_args()

    # 1 · Acquire
    tweepfake_df = load_tweepfake()
    news_df = load_fake_news()
    custom_df = generate_custom_synthetic(args.custom_n)

    # 2 · Merge
    df = pd.concat([tweepfake_df, news_df, custom_df], ignore_index=True)
    print(f"\n   Total raw samples: {len(df)}")

    # 3 · Clean
    df = clean_dataframe(df)

    # 4 · Features
    if args.skip_perplexity:
        print("[5/6] Skipping perplexity (--skip-perplexity flag) ...")
        df["burstiness"] = df["text"].apply(compute_burstiness)
        df["perplexity"] = 0.0
    else:
        df = add_features(df)

    # 5 · Split & save
    split_and_save(df, DATA_DIR)

    print("\n[OK] Data pipeline complete!")


if __name__ == "__main__":
    main()
