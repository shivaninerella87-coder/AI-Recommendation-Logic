# AI Recommendation Logic: Tech Stack Recommender

## 📌 Project Description

This project is a **Digital Matchmaker Engine** designed to solve "Choice Overload" by mapping a user's raw skills and career goals to specific job roles and technology stacks. Moving beyond passive data classification, this system focuses on **Active Prediction**.

Instead of relying on community behavior (Collaborative Filtering), this engine is built entirely on **Content-Based Filtering**. It uses pure similarity logic and mathematical vector mapping to transform qualitative user inputs into a structured, objective Top-N recommendation list.

## 🧠 Core Pipeline Architecture

The system is built on a strict 4-step Input-Process-Output (IPO) pipeline:

1. **Ingestion:** Captures the "user state" by taking a minimum of three explicit skills as input to ensure data density.
2. **Scoring:** Transforms both the dataset and the user profile into mathematical arrays, then applies similarity math to score the alignment between the user and every available item.
3. **Sorting:** Organizes the processed dataset in descending order based on the calculated scores.
4. **Filtering:** Truncates the final output to generate a Top-N list (e.g., Top 3 paths), presenting only the highest-scoring matches to prevent information overload.

## 📖 Key Terminology & Concepts

To build this engine, several foundational Artificial Intelligence and Machine Learning concepts were utilized:

### Algorithm Design

* **Content-Based Filtering:** A recommendation method driven by item attributes, mapping user preferences directly to the intrinsic properties of the items themselves, independent of other users' behavior.
* **Vector Mapping:** The process of translating qualitative raw data (like words) into numerical arrays within a shared vocabulary space so the machine can process them.
* **The Cold Start Problem:** The challenge a system faces when it encounters a brand-new user (vector of zeros) or a newly added item without enough historical data to execute its similarity math.

### Mathematical Logic

* **TF-IDF (Term Frequency-Inverse Document Frequency):** A statistical weighting method used for feature extraction. It rewards highly specific, descriptive tags and penalizes high-frequency, generic words (like "the" or "software").
* **Cosine Similarity:** The industry-standard metric used to measure the mathematical angle (orientation) between two vectors in a multidimensional space. It ensures the engine evaluates the *alignment* of interests rather than just the absolute size/magnitude of a profile.
* **Euclidean Distance & Jaccard Similarity:** Alternative classification metrics that evaluate straight-line distance and binary overlap, respectively. (These were bypassed in this project as they lack the nuance required for unstructured text environments).

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**
* `pandas` (for data ingestion and structure)
* `scikit-learn` (for `TfidfVectorizer` and `cosine_similarity` computations)


* **Dataset:** Custom `raw_skills.csv` mapping job roles to specific technologies.
