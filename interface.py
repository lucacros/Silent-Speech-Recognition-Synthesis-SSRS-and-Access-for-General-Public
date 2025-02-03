import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import random
import threading
import time
import pyaudio
import wave
import os


class AudioPromptApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Prompt Application")

        # UI Components
        self.word_label = tk.Label(root, text="", font=("Helvetica", 24), fg="green")
        self.word_label.pack(pady=20)

        self.start_button = tk.Button(root, text="Start Session", command=self.start_session)
        self.start_button.pack(pady=10)

        self.progress_label = tk.Label(root, text="Progress: 0/0", font=("Helvetica", 14))
        self.progress_label.pack(pady=10)

        self.device_label = tk.Label(root, text="Select Input Device:", font=("Helvetica", 14))
        self.device_label.pack(pady=10)

        self.device_listbox = tk.Listbox(root, height=5)
        self.device_listbox.pack(pady=10)

        self.refresh_button = tk.Button(root, text="Refresh Devices", command=self.refresh_device_list)
        self.refresh_button.pack(pady=10)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line_left, = self.ax.plot([], [], label="Left Channel")
        self.line_right, = self.ax.plot([], [], label="Right Channel")
        self.ax.set_xlim(0, 1024)
        self.ax.set_ylim(-32768, 32767)
        self.ax.legend()
        self.ax.set_title("Audio Signal (Stereo)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        # Initialize variables
        self.words = ["Hello", "Hi", "Thankyou", "Goodbye", "Please", "Yes", "No", "Okay", "Sorry", "GoodMorning", "Bye", "Cheers", "Congratulations", "Excuseme", "Later"]
        self.current_word = None
        self.progress = 0
        self.total_words = 100
        self.running = False
        self.selected_device_index = None
        self.audio_thread = None

        # Audio settings
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 2  # Stereo
        self.rate = 44100  # Sampling rate
        self.audio_stream = None

        # Output folder
        self.output_folder = "Test_session"
        os.makedirs(self.output_folder, exist_ok=True)

        self.refresh_device_list()

    def refresh_device_list(self):
        self.device_listbox.delete(0, tk.END)
        audio = pyaudio.PyAudio()
        try:
            for i in range(audio.get_device_count()):
                device = audio.get_device_info_by_index(i)
                self.device_listbox.insert(tk.END, f"{i}: {device['name']}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to list devices: {e}")
        finally:
            audio.terminate()

    def start_session(self):
        selected = self.device_listbox.curselection()
        if not selected:
            messagebox.showerror("Error", "Please select an input device!")
            return

        self.selected_device_index = int(self.device_listbox.get(selected[0]).split(":")[0])
        self.progress = 0
        self.running = True
        threading.Thread(target=self.capture_and_plot_audio_continuously, daemon=True).start()
        threading.Thread(target=self.run_session, daemon=True).start()

    def run_session(self):
        audio = pyaudio.PyAudio()
        try:
            self.audio_stream = audio.open(format=self.format,
                                           channels=self.channels,
                                           rate=self.rate,
                                           input=True,
                                           input_device_index=self.selected_device_index,
                                           frames_per_buffer=self.chunk_size)

            for i in range(self.total_words):
                if not self.running:
                    break

                self.current_word = random.choice(self.words)
                self.word_label.config(text=self.current_word)
                self.progress_label.config(text=f"Progress: {self.progress + 1}/{self.total_words}")

                # Capture audio for the current word
                frames = []
                recording_duration = 2  # Duration in seconds
                num_chunks = int(self.rate / self.chunk_size * recording_duration)

                for _ in range(num_chunks):
                    if not self.running:
                        break
                    audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(audio_data)

                # Save the audio to a WAV file
                self.save_audio(frames, self.current_word, i + 1)

                self.progress += 1
                time.sleep(0.5)  # Pause before moving to the next word

            self.word_label.config(text="Session Complete!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during the session: {e}")
            self.running = False

        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            audio.terminate()

    def save_audio(self, frames, word, index):
        """Saves audio data to a WAV file."""
        filename = os.path.join(self.output_folder, f"{word}_{index}.wav")
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(frames))
            print(f"Audio saved: {filename}")
        except Exception as e:
            print(f"Failed to save audio: {e}")

    def capture_and_plot_audio_continuously(self):
        audio = pyaudio.PyAudio()
        try:
            self.audio_stream = audio.open(format=self.format,
                                           channels=self.channels,
                                           rate=self.rate,
                                           input=True,
                                           input_device_index=self.selected_device_index,
                                           frames_per_buffer=self.chunk_size)

            while self.running:
                try:
                    audio_data = self.audio_stream.read(self.chunk_size, exception_on_overflow=False)
                    signal = np.frombuffer(audio_data, dtype=np.int16)

                    # Separate left and right channels
                    left_channel = signal[0::2]
                    right_channel = signal[1::2]

                    # Update the plot
                    self.line_left.set_data(range(len(left_channel)), left_channel)
                    self.line_right.set_data(range(len(right_channel)), right_channel)

                    self.ax.set_xlim(0, len(left_channel))
                    self.canvas.draw()
                except Exception as e:
                    print(f"Error in audio capture: {e}")
                    break

        except Exception as e:
            print(f"Error initializing audio capture: {e}")

        finally:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            audio.terminate()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioPromptApp(root)
    root.mainloop()
