# Movie-Llama: A Movie Recommendation Assistant



Movie-Llama is an AI-driven movie recommendation assistant designed to spotlight films from underrepresented cultures and communities. Leveraging a Large Language Model (LLM) and vector embeddings, Movie-Llama provides users with curated movie suggestions that celebrate cultural diversity in global cinema. Built using data from The Movie Database (TMDB), this assistant enhances user discovery of niche and diverse films, promoting cross-cultural understanding and appreciation.

## Project Background

In a media landscape often dominated by mainstream films, many unique cultural voices go unheard. Movie-Llama aims to counter this by focusing on culturally rich films from smaller, underrepresented communities. By doing so, it helps reduce media homogenization, enriches cultural awareness, and connects users with films they might not otherwise discover.

## Features

- **Personalized Recommendations**: Tailored movie suggestions based on user queries related to genre, language, and cultural context.
- **Diverse Film Selection**: Emphasizes films from smaller communities and diverse cultures.
- **Efficient Retrieval and Generation**: Combines retrieval-augmented generation with vector embedding techniques for relevant and engaging recommendations.

## Getting Started

### Prerequisites

To run this project, you’ll need:

- Python 3.8 or above
- [TMDB API Key](https://www.themoviedb.org/settings/api) (sign up for an API key on TMDB)
- Libraries: `nomic`, `chroma`, `transformers`

### Installation

Clone the repository:

```bash
git clone https://github.com/drunken-monkey-ops/Movie-Llama.git
cd Movie-Llama
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up your environment variables for the TMDB API key:

```bash
export TMDB_API_KEY=your_api_key
```

### Usage

1. **Data Extraction**: The assistant extracts movie data from TMDB, focusing on titles, genres, original languages, and summaries.
2. **Embedding Creation**: Movie data is converted into vector embeddings using the `nomic-embed-text` model, creating a searchable vector space.
3. **Recommendation Query**: Users can query the assistant (e.g., "Recommend movies from Indian cinema"), and the assistant will search for relevant movie vectors.
4. **Natural Language Response**: The LLM generates coherent recommendations, highlighting cultural relevance in a user-friendly format.

Run the assistant:

```bash
python movie_llama.py
```

## Methodology

1. **Data Processing**: Movie data is fetched and structured using JSON format, then split into manageable text chunks.
2. **Vector Embeddings**: Each movie is represented as a vector using `nomic-embed-text`, enabling semantic similarity search.
3. **Vector Storage and Retrieval**: Embeddings are stored in a vector database (Chroma), and user queries retrieve similar movie vectors.
4. **LLM Generation**: Retrieved data is passed to the LLM (LLama 3.1) to produce natural language recommendations.

## Social Impact

Movie-Llama promotes inclusivity and cultural diversity by recommending films from various cultural backgrounds. Its approach helps preserve language diversity, support independent filmmakers, and broaden the scope of media consumption.

## Contributing

Contributions are welcome! Please submit a pull request or raise an issue for any suggestions.

## License

Distributed under the MIT License. See `LICENSE` for more information.



## Acknowledgments

- [TMDB](https://www.themoviedb.org/) for the data API
- [Ollama Embeddings Documentation](https://ollama.com/) and [Meta AI's LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) for 
