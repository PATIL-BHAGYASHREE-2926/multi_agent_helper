# Day 4 Build Spec — Naive RAG Project
### Live Code Walkthrough Demo

---

## What This Is

A real, working naive RAG project — indexing a document, embedding it, storing it in a vector DB, and answering queries by retrieving relevant chunks and generating a grounded response. Built to be traced live against the RAG pipeline diagram taught earlier in the day, so every piece of code has an obvious "this is step X from the diagram" moment.

This is **not** a toy or oversimplified snippet — it should be a genuinely complete, runnable naive RAG pipeline, simple enough to walk through live but real enough to demonstrate how the pieces actually connect.

---

## Where It's Used

- **Day 4, Part 6 — Live Code Walkthrough (Slides 22–25)**: the core hands-on segment of the day
- Directly follows **Part 4 (The RAG Pipeline, Slides 13–16)** — the diagram taught there is the map this project is traced against
- Directly follows **Part 5 (Types of RAG, Slides 17–21)** — this project is explicitly the "naive RAG" example named there

---

## What It Must Contain

### 1. Indexing pipeline (runs once, before the live query)
- **Document ingestion** — load a source document (text file, markdown, or similar)
- **Chunking** — split the document into chunks (simple/fixed-size is fine here — full chunking strategy depth is Day 5's job, not this project's)
- **Embedding** — generate an embedding vector for each chunk
- **Storage** — store chunks + embeddings in a vector DB

### 2. Query pipeline (runs live, on stage)
- **Query embedding** — embed the user's live query the same way chunks were embedded
- **Retrieval** — query the vector DB for the top-k most similar chunks
- **Context injection** — insert retrieved chunks into the prompt sent to the model
- **Generation** — model produces an answer grounded in the retrieved context

### 3. Code must map cleanly onto the pipeline diagram (Day 4 Slide 13)
Every function/block should correspond to one labeled box:
- `load_document()` → Document ingestion
- `chunk_document()` → Chunking
- `embed_chunks()` → Embedding
- `store_in_vector_db()` → Storage
- `embed_query()` → Query embedding
- `retrieve_top_k()` → Retrieval
- `build_prompt_with_context()` → Context injection
- `generate_answer()` → Generation

Naming functions to match the diagram's vocabulary (not generic names) makes the live trace-through effortless — you point at a box on the diagram, then point at the matching function.

### 4. Source document
- A short, clean document with **clear, verifiable facts** — something where a retrieved answer is obviously either right or wrong to the audience
- Topic-agnostic, but should be something the audience can follow without domain expertise
- Length: short enough to index quickly live if needed, long enough to require real chunking (a few hundred to ~1000 words is reasonable)
- *(Note: keep this document separate from Day 5's synthetic "designed to break" document — this one should chunk and retrieve cleanly, since Day 4 is about RAG working, not RAG failing)*

### 5. The live query
- One prepared query that clearly demonstrates retrieval working — the retrieved chunk(s) should be obviously relevant, and the final answer should be clearly grounded in them, not the model's general knowledge
- Ideally, a question whose answer the model would likely get wrong or vague *without* retrieval, so the "grounded in context" value is visible by contrast

---

## What the End Output Should "Feel Like" On Stage

1. **Indexing walkthrough** — show chunking happening (print chunk count/sizes), show an embedding call return a vector, show storage confirm the write. Fast, not belabored — this part is about recognition ("oh, that's the chunking box"), not suspense.
2. **Query walkthrough** — show the query get embedded, show retrieval return its top-k chunks **printed to the screen** (so the audience can read what was retrieved, not just trust it happened), show the final prompt assembled with that context, then the generated answer.
3. The answer should visibly and specifically draw on the retrieved content — ideally quote or closely paraphrase something only present in the source document, proving the model used retrieval rather than its own general knowledge.

---

## Build Requirements

- **Vector DB**: any simple, easy-to-run option (a lightweight local vector store is fine — doesn't need to be a production-grade managed service for a 2–3hr workshop demo)
- **Embedding model + generation model**: your choice of provider — keep consistent with whatever's used elsewhere in the track (Groq/Gemini) where possible, for audience familiarity
- **API keys**: loaded from environment variables, never hardcoded or visible on screen
- **Print statements at every stage** — this project's value on stage comes from *visibility*, not just a correct final answer. Don't let any step run silently.
- **Tested end-to-end before the session** — both indexing and query pipelines confirmed working, on venue-equivalent network conditions
- **Backup plan**: a saved run (terminal output or screenshot) in case live API calls fail mid-session

---

## Status
- [x] Source document chosen/written (short, clear facts, clean retrieval)
- [x] Indexing pipeline built and tested (chunk → embed → store)
- [x] Query pipeline built and tested (embed → retrieve → inject → generate)
- [x] Function names mapped explicitly to pipeline diagram vocabulary
- [x] Live query selected and confirmed to retrieve the right chunk(s) and produce a clearly grounded answer
- [x] Print/output statements added at every stage for visibility
- [x] Tested on venue-equivalent network conditions
- [x] Backup output captured and saved
- [x] API keys moved to environment variables, confirmed not visible on screen
