import streamlit as st
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import librosa
import numpy as np
import soundfile as sf
import io
import tempfile
import os
from pathlib import Path
import sys

# Set page configuration
st.set_page_config(
    page_title="Creative Studio - Image & Audio AI",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'tool' not in st.session_state:
    st.session_state.tool = 'image'

# Create a unified creative studio
class CreativeStudio:
    def __init__(self):
        """Initialize the creative studio with both image and audio tools"""
        self.image_editor = None
        self.audio_remasterer = None
        
    def init_image_editor(self):
        """Initialize the image editor if not already done"""
        if self.image_editor is None:
            self.image_editor = ProImageEditor()
        return self.image_editor
    
    def init_audio_remasterer(self):
        """Initialize the audio remasterer if not already done"""
        if self.audio_remasterer is None:
            self.audio_remasterer = AudioRemaster()
        return self.audio_remasterer

# Image Editor Class
class ProImageEditor:
    def __init__(self):
        """Initialize the image editor with AI models"""
        self.model_id = "timbrooks/instruct-pix2pix"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        
    def load_model(self):
        """Load the AI model with progress indication"""
        with st.spinner("Loading professional AI models (this may take a minute)..."):
            if self.pipe is None:
                self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None
                ).to(self.device)
            st.success("✅ Professional AI editor ready!")
    
    def edit_image(self, image: Image.Image, prompt: str, resolution: int = 512, 
                  num_inference_steps: int = 20, image_guidance_scale: float = 1.5, 
                  text_guidance_scale: float = 7.5) -> Image.Image:
        """Edit image based on text prompt using professional AI"""
        if self.pipe is None:
            self.load_model()
        
        # Resize image while maintaining aspect ratio
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            new_width = resolution
            new_height = int(resolution / aspect_ratio)
        else:
            new_height = resolution
            new_width = int(resolution * aspect_ratio)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Generate edited image
        with torch.no_grad():
            edited_image = self.pipe(
                prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=text_guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]
        
        return edited_image

# Audio Remasterer Class
class AudioRemaster:
    def __init__(self):
        """Initialize the audio remastering engine"""
        self.sample_rate = 44100
        self.supported_formats = ['.mp3', '.wav', '.flac', '.m4a', '.ogg']
        
    def load_audio(self, file_bytes):
        """Load audio file from bytes"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            # Load with librosa
            audio, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=False)
            
            # Clean up
            os.unlink(tmp_path)
            
            return audio, sr
        except Exception as e:
            st.error(f"Error loading audio: {str(e)}")
            return None, None
    
    def reduce_noise(self, audio, sr, stationary=True, prop_decrease=0.8):
        """Apply AI-powered noise reduction"""
        try:
            import noisereduce as nr
            if len(audio.shape) == 1:
                # Mono audio
                reduced_noise = nr.reduce_noise(y=audio, sr=sr, stationary=stationary, prop_decrease=prop_decrease)
            else:
                # Stereo audio - process each channel
                reduced_noise = np.zeros_like(audio)
                for channel in range(audio.shape[0]):
                    reduced_noise[channel] = nr.reduce_noise(
                        y=audio[channel], sr=sr, stationary=stationary, prop_decrease=prop_decrease
                    )
            return reduced_noise
        except Exception as e:
            st.error(f"Noise reduction failed: {str(e)}")
            return audio
    
    def enhance_clarity(self, audio, sr, high_pass=80, low_pass=8000):
        """Enhance vocal clarity using EQ"""
        try:
            # Apply high-pass filter to remove low-frequency rumble
            audio_filtered = librosa.effects.preemphasis(audio, coef=0.97)
            
            # Apply low-pass filter to reduce harsh highs
            if len(audio_filtered.shape) == 1:
                audio_filtered = librosa.effects.lowpass(audio_filtered, cutoff=low_pass)
                audio_filtered = librosa.effects.highpass(audio_filtered, cutoff=high_pass)
            else:
                for channel in range(audio_filtered.shape[0]):
                    audio_filtered[channel] = librosa.effects.lowpass(audio_filtered[channel], cutoff=low_pass)
                    audio_filtered[channel] = librosa.effects.highpass(audio_filtered[channel], cutoff=high_pass)
            
            return audio_filtered
        except Exception as e:
            st.error(f"Clarity enhancement failed: {str(e)}")
            return audio
    
    def add_spatial_depth(self, audio, sr, width=1.5):
        """Add stereo widening and spatial depth"""
        try:
            if len(audio.shape) == 1:
                # Convert mono to stereo
                audio = np.vstack([audio, audio])
            
            # Apply subtle stereo widening
            mid = (audio[0] + audio[1]) / 2
            side = (audio[0] - audio[1]) / 2
            
            # Enhance side channel for width
            side_enhanced = side * width
            
            # Reconstruct stereo signal
            left = mid + side_enhanced
            right = mid - side_enhanced
            
            # Normalize to prevent clipping
            max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
            if max_val > 1.0:
                left /= max_val
                right /= max_val
            
            return np.vstack([left, right])
        except Exception as e:
            st.error(f"Spatial enhancement failed: {str(e)}")
            return audio
    
    def dynamic_range_compression(self, audio, sr, threshold=0.5, ratio=4.0, attack=0.003, release=0.1):
        """Apply dynamic range compression for consistent volume"""
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize
            
            # Convert to PyDub AudioSegment for processing
            if len(audio.shape) == 1:
                audio_segment = AudioSegment(
                    audio.tobytes(), 
                    frame_rate=sr,
                    sample_width=audio.dtype.itemsize,
                    channels=1
                )
            else:
                audio_segment = AudioSegment(
                    audio.T.tobytes(), 
                    frame_rate=sr,
                    sample_width=audio.dtype.itemsize,
                    channels=2
                )
            
            # Apply compression
            compressed = audio_segment.compress_dynamic_range(
                threshold=threshold,
                ratio=ratio,
                attack=attack,
                release=release
            )
            
            # Convert back to numpy array
            samples = np.array(compressed.get_array_of_samples())
            if compressed.channels == 2:
                samples = samples.reshape((-1, 2)).T
            else:
                samples = samples.reshape(1, -1)
            
            return samples.astype(np.float32) / 32768.0
        except Exception as e:
            st.error(f"Compression failed: {str(e)}")
            return audio
    
    def master_audio(self, audio, sr, target_lufs=-14.0):
        """Apply final mastering with loudness normalization"""
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize
            
            # Convert to PyDub for loudness normalization
            if len(audio.shape) == 1:
                audio_segment = AudioSegment(
                    (audio * 32768).astype(np.int16).tobytes(), 
                    frame_rate=sr,
                    sample_width=2,
                    channels=1
                )
            else:
                audio_segment = AudioSegment(
                    (audio.T * 32768).astype(np.int16).tobytes(), 
                    frame_rate=sr,
                    sample_width=2,
                    channels=2
                )
            
            # Normalize to target loudness
            normalized = normalize(audio_segment)
            
            # Convert back to numpy array
            samples = np.array(normalized.get_array_of_samples())
            if normalized.channels == 2:
                samples = samples.reshape((-1, 2)).T
            else:
                samples = samples.reshape(1, -1)
            
            return samples.astype(np.float32) / 32768.0
        except Exception as e:
            st.error(f"Mastering failed: {str(e)}")
            return audio
    
    def process_audio(self, file_bytes, options):
        """Process audio with selected remastering options"""
        # Load audio
        audio, sr = self.load_audio(file_bytes)
        if audio is None:
            return None
        
        with st.spinner("🎚️ Applying AI remastering..."):
            # Apply selected enhancements
            if options['noise_reduction']:
                audio = self.reduce_noise(audio, sr, stationary=options['stationary_noise'])
            
            if options['clarity_enhancement']:
                audio = self.enhance_clarity(
                    audio, sr, 
                    high_pass=options['high_pass'],
                    low_pass=options['low_pass']
                )
            
            if options['spatial_enhancement']:
                audio = self.add_spatial_depth(audio, sr, width=options['stereo_width'])
            
            if options['compression']:
                audio = self.dynamic_range_compression(
                    audio, sr,
                    threshold=options['threshold'],
                    ratio=options['ratio']
                )
            
            # Always apply final mastering
            audio = self.master_audio(audio, sr, target_lufs=options['target_lufs'])
        
        return audio, sr

# Main application
def main():
    # Initialize the creative studio
    studio = CreativeStudio()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #FF6B6B;
            text-align: center;
            margin-bottom: 1rem;
        }
        .tool-selector {
            display: flex;
            justify-content: center;
            margin-bottom: 2rem;
        }
        .tool-button {
            padding: 0.5rem 1.5rem;
            margin: 0 0.5rem;
            border-radius: 0.5rem;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        .tool-button.active {
            background-color: #FF6B6B;
            color: white;
        }
        .tool-button:not(.active) {
            background-color: #f0f2f6;
            color: #262730;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎨 Creative Studio</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Professional AI tools for image editing and audio remastering</p>', unsafe_allow_html=True)
    
    # Tool selector
    tool = st.radio(
        "Select Tool",
        ["🖼️ Image Editor", "🎵 Audio Remaster"],
        horizontal=True,
        key="tool_selector"
    )
    
    if tool == "🖼️ Image Editor":
        st.session_state.tool = 'image'
    else:
        st.session_state.tool = 'audio'
    
    # Display the selected tool
    if st.session_state.tool == 'image':
        display_image_editor(studio)
    else:
        display_audio_remasterer(studio)

def display_image_editor(studio):
    """Display the image editor interface"""
    st.header("🖼️ Professional AI Image Editor")
    st.markdown("Transform your images with natural language instructions")
    
    # Initialize editor
    editor = studio.init_image_editor()
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Original Image")
        uploaded_file = st.file_uploader(
            "Upload an image to edit",
            type=["jpg", "jpeg", "png", "webp"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_column_width=True)
            
            with st.expander("Image Details"):
                st.write(f"Dimensions: {image.width} × {image.height} pixels")
                st.write(f"Mode: {image.mode}")
    
    with col2:
        st.subheader("✨ Edited Image")
        prompt = st.text_area(
            "Describe your desired edit",
            placeholder="e.g., 'Turn this into a watercolor painting', 'Add a snowy background'",
            height=100
        )
        
        # Professional settings
        with st.expander("Professional Settings"):
            resolution = st.slider("Resolution Quality", 256, 1024, 512, 128)
            num_steps = st.slider("Editing Precision", 10, 50, 20, 5)
            image_fidelity = st.slider("Original Image Fidelity", 0.5, 2.0, 1.5, 0.1)
            prompt_strength = st.slider("Prompt Adherence", 5.0, 15.0, 7.5, 0.5)
        
        edit_button = st.button(
            "🚀 Apply Professional Edit",
            type="primary",
            disabled=uploaded_file is None or not prompt.strip()
        )
        
        if edit_button and uploaded_file and prompt.strip():
            with st.spinner("🎨 Applying professional edits..."):
                try:
                    edited_image = editor.edit_image(
                        image=image,
                        prompt=prompt,
                        resolution=resolution,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_fidelity,
                        text_guidance_scale=prompt_strength
                    )
                    
                    st.image(edited_image, caption="Edited Image", use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    edited_image.save(buf, format="JPEG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="💾 Download Edited Image",
                        data=byte_im,
                        file_name="edited_image.jpg",
                        mime="image/jpeg"
                    )
                    
                    st.success("✅ Professional edit completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during editing: {str(e)}")
                    st.info("Try reducing resolution or simplifying your prompt.")
    
    # Example prompts
    st.markdown("---")
    st.subheader("💡 Example Prompts")
    
    example_prompts = [
        ("🎨 Artistic Styles", [
            "Turn into a watercolor painting",
            "Make it look like a Van Gogh",
            "Convert to charcoal sketch",
            "Apply anime style",
            "Make it look like a Renaissance painting"
        ]),
        ("🌅 Scene & Environment", [
            "Add a sunset background",
            "Turn day into night",
            "Add snow to the scene",
            "Place in a tropical beach",
            "Add autumn leaves"
        ]),
        ("👥 People & Portraits", [
            "Make me look younger",
            "Add a professional smile",
            "Change hair color to blonde",
            "Add sunglasses",
            "Remove wrinkles"
        ]),
        ("🏗️ Objects & Items", [
            "Add a vintage car",
            "Replace with modern furniture",
            "Add a bouquet of flowers",
            "Insert a laptop",
            "Add a bookshelf"
        ]),
        ("📸 Photo Effects", [
            "Make it look vintage",
            "Apply black and white",
            "Add film grain",
            "Make it look like a polaroid",
            "Add lens flare"
        ])
    ]
    
    cols = st.columns(len(example_prompts))
    
    for i, (category, prompts) in enumerate(example_prompts):
        with cols[i]:
            st.markdown(f"#### {category}")
            for prompt in prompts:
                st.markdown(f"- {prompt}")

def display_audio_remasterer(studio):
    """Display the audio remasterer interface"""
    st.header("🎵 Professional AI Audio Remaster")
    st.markdown("Transform your audio tracks with studio-quality enhancement")
    
    # Initialize remasterer
    remasterer = studio.init_audio_remasterer()
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Audio")
        uploaded_file = st.file_uploader(
            "Upload an audio file to remaster",
            type=['mp3', 'wav', 'flac', 'm4a', 'ogg']
        )
        
        if uploaded_file is not None:
            # Display audio player
            st.audio(uploaded_file, format='audio/mp3')
            
            # Show file info
            file_details = {
                "Filename": uploaded_file.name,
                "Size": f"{len(uploaded_file.getvalue()) / 1024 / 1024:.2f} MB",
                "Type": uploaded_file.type
            }
            
            with st.expander("File Details"):
                for key, value in file_details.items():
                    st.write(f"**{key}**: {value}")
    
    with col2:
        st.subheader("✨ Remastered Audio")
        
        if uploaded_file is not None:
            # Professional settings
            with st.expander("Remastering Controls"):
                # Noise Reduction
                noise_reduction = st.checkbox("Enable Noise Reduction", value=True)
                stationary_noise = st.checkbox("Stationary Noise (hiss, hum)", value=True)
                
                # Clarity Enhancement
                clarity_enhancement = st.checkbox("Enhance Vocal Clarity", value=True)
                if clarity_enhancement:
                    high_pass = st.slider("High-Pass Filter (Hz)", 20, 200, 80, 10)
                    low_pass = st.slider("Low-Pass Filter (Hz)", 2000, 16000, 8000, 500)
                
                # Spatial Enhancement
                spatial_enhancement = st.checkbox("Add Stereo Width", value=True)
                if spatial_enhancement:
                    stereo_width = st.slider("Stereo Width", 1.0, 3.0, 1.5, 0.1)
                
                # Dynamic Range
                compression = st.checkbox("Apply Compression", value=True)
                if compression:
                    threshold = st.slider("Threshold (dB)", -40, 0, -20, 1)
                    ratio = st.slider("Ratio", 1.0, 10.0, 4.0, 0.5)
                
                # Mastering
                target_lufs = st.slider("Target Loudness (LUFS)", -20, -8, -14, 1)
            
            # Create options dictionary
            options = {
                'noise_reduction': noise_reduction,
                'stationary_noise': stationary_noise,
                'clarity_enhancement': clarity_enhancement,
                'high_pass': high_pass if clarity_enhancement else 80,
                'low_pass': low_pass if clarity_enhancement else 8000,
                'spatial_enhancement': spatial_enhancement,
                'stereo_width': stereo_width if spatial_enhancement else 1.5,
                'compression': compression,
                'threshold': threshold if compression else -20,
                'ratio': ratio if compression else 4.0,
                'target_lufs': target_lufs
            }
            
            # Process button
            process_button = st.button(
                "🚀 Apply AI Remastering",
                type="primary"
            )
            
            if process_button:
                # Process audio
                processed_audio, sr = remasterer.process_audio(uploaded_file.getvalue(), options)
                
                if processed_audio is not None:
                    # Save processed audio to bytes
                    with io.BytesIO() as buffer:
                        if len(processed_audio.shape) == 1:
                            sf.write(buffer, processed_audio, sr, format='WAV')
                        else:
                            sf.write(buffer, processed_audio.T, sr, format='WAV')
                        buffer.seek(0)
                        
                        # Display audio player
                        st.audio(buffer, format='audio/wav')
                        
                        # Download button
                        st.download_button(
                            label="💾 Download Remastered Audio",
                            data=buffer,
                            file_name=f"remastered_{uploaded_file.name}",
                            mime="audio/wav"
                        )
                    
                    st.success("✅ Audio remastered successfully!")
                    
                    # Show processing summary
                    with st.expander("Processing Summary"):
                        st.write("Applied enhancements:")
                        if noise_reduction:
                            st.write("- ✅ Noise reduction")
                        if clarity_enhancement:
                            st.write("- ✅ Vocal clarity enhancement")
                        if spatial_enhancement:
                            st.write("- ✅ Stereo widening")
                        if compression:
                            st.write("- ✅ Dynamic range compression")
                        st.write("- ✅ Loudness normalization")
    
    # Use cases
    st.markdown("---")
    st.subheader("🎯 Perfect For...")
    
    use_cases = [
        ("🎤 Podcasts", "Remove background noise and enhance vocal clarity"),
        ("🎵 Music", "Add depth and professional mastering to your tracks"),
        ("📼 Recordings", "Restore old recordings and reduce tape hiss"),
        ("🎧 Streaming", "Optimize audio for online platforms"),
        ("📱 Mobile", "Improve audio quality for mobile playback")
    ]
    
    cols = st.columns(len(use_cases))
    
    for i, (title, description) in enumerate(use_cases):
        with cols[i]:
            st.markdown(f"#### {title}")
            st.write(description)

if __name__ == "__main__":
    main()