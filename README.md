# 🎨 AI Art Generator  

An interactive **AI-powered art creation and editing tool** built with **Streamlit**, **Hugging Face APIs**, and **CLIP embeddings**.  
This app allows users to:  
- Generate unique artworks from **text prompts**.  
- Apply **artistic styles** automatically or manually.  
- Refine results with **natural language feedback**.  
- Explore **variations** and co-create iteratively.  

---

## 🚀 Features
- **Prompt-to-Art Generation** → Create artworks with Stable Diffusion, Flux, or SD-XL.  
- **Style Selector Agent** → Extracts styles, subjects, and descriptors for better prompts.  
- **IR Agent (Image Retrieval)** → Finds reference artworks with CLIP embeddings.  
- **Feedback Generator Agent** → Modify images with feedback like *“make it brighter”*, *“add glasses”*, or *“remove background”*.  
- **Subscription System** → Free plan (with watermark) or Premium (no watermark).  
- **History Tracking** → Saves generated image metadata for reuse.  

---

## 🧩 Agent Architecture  

The app is designed as a **multi-agent system**:  

1. **IR Agent**  
   - Retrieves **reference artworks** using CLIP similarity search.  
   - Provides grounding for consistent style.  

2. **Style Selector Agent**  
   - Extracts **nouns/adjectives** from prompts with spaCy.  
   - Chooses or auto-detects **artistic styles**.  
   - Builds **refined prompts**.  

3. **Image Synthesizer Agent**  
   - Generates final images using Hugging Face Inference API.  
   - Supports text-to-image, image-to-image, and local inpainting.  
   - Applies watermarks (Free) or produces clean outputs (Premium).  

4. **Feedback Generator Agent**  
   - Applies **direct adjustments** (brightness, color, contrast, filters).  
   - Performs **object modifications** (add/remove hats, sky, background).  
   - Understands **natural language feedback** for iterative refinement.  

---



