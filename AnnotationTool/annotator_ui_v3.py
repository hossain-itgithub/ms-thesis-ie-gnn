import os
import json
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog, Toplevel, colorchooser
from PIL import Image, ImageTk, ImageDraw, ExifTags
from googletrans import Translator

class Annotator:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Annotator")
        self.master.geometry("1800x1000")

        self.image_dir = filedialog.askdirectory(title="Select Image Directory")
        self.ocr_dir = filedialog.askdirectory(title="Select OCR Data Directory")
        self.save_dir = filedialog.askdirectory(title="Select Save Directory")

        self.unusable_files_path = os.path.join(self.save_dir, "unusable_files.txt")
        self.unusable_files = self.load_unusable_files()

        self.exclude_annotated = BooleanVar()
        self.exclude_annotated.set(False)

        self.exclude_unusable = BooleanVar()
        self.exclude_unusable.set(True)

        self.checkbox_frame = Frame(master)
        self.checkbox_frame.pack(side=TOP, padx=10, pady=10)
        self.exclude_checkbox = Checkbutton(self.checkbox_frame, text="Exclude images with annotations", variable=self.exclude_annotated, command=self.update_image_list)
        self.exclude_checkbox.pack(side=LEFT, padx=10, pady=10)

        self.exclude_unusable_checkbox = Checkbutton(self.checkbox_frame, text="Exclude unusable files", variable=self.exclude_unusable, command=self.update_image_list)
        self.exclude_unusable_checkbox.pack(side=LEFT, padx=10, pady=10)

        self.track_unusable_button = Button(self.checkbox_frame, text="Track Unusable File", command=self.track_unusable_file)
        self.track_unusable_button.pack(side=LEFT, padx=10, pady=10)

        self.image_files = []
        self.update_image_list()
        
        self.current_image_index = 0
        self.ocr_data = []
        self.selected_boxes = []

        self.class_colors = {
            "company": "green",
            "date": "yellow",
            "address": "blue",
            "total": "purple"
        }
        self.default_box_color = "red"

        self.image_label = Label(master)
        self.image_label.pack(side=LEFT, padx=10, pady=10)
        
        self.ocr_listbox = Listbox(master, selectmode=MULTIPLE, width=50, height=40)
        self.ocr_listbox.pack(side=LEFT, padx=10, pady=10)
        
        self.annotation_frame = Frame(master)
        self.annotation_frame.pack(side=LEFT, padx=10, pady=10)
        
        Label(self.annotation_frame, text="Company:").grid(row=0, column=0, sticky=W)
        self.company_entry = Entry(self.annotation_frame, width=40)
        self.company_entry.grid(row=0, column=1, pady=5)
        self.company_button = Button(self.annotation_frame, text="Copy", command=lambda: self.copy_to_entry(self.company_entry))
        self.company_button.grid(row=0, column=2, padx=5)
        self.company_color_button = Button(self.annotation_frame, text="Color", command=lambda: self.choose_color("company"))
        self.company_color_button.grid(row=0, column=3, padx=5)
        
        Label(self.annotation_frame, text="Date:").grid(row=1, column=0, sticky=W)
        self.date_entry = Entry(self.annotation_frame, width=40)
        self.date_entry.grid(row=1, column=1, pady=5)
        self.date_button = Button(self.annotation_frame, text="Copy", command=lambda: self.copy_to_entry(self.date_entry))
        self.date_button.grid(row=1, column=2, padx=5)
        self.date_color_button = Button(self.annotation_frame, text="Color", command=lambda: self.choose_color("date"))
        self.date_color_button.grid(row=1, column=3, padx=5)
        
        Label(self.annotation_frame, text="Address:").grid(row=2, column=0, sticky=W)
        self.address_entry = Entry(self.annotation_frame, width=40)
        self.address_entry.grid(row=2, column=1, pady=5)
        self.address_button = Button(self.annotation_frame, text="Copy", command=lambda: self.copy_to_entry(self.address_entry))
        self.address_button.grid(row=2, column=2, padx=5)
        self.address_color_button = Button(self.annotation_frame, text="Color", command=lambda: self.choose_color("address"))
        self.address_color_button.grid(row=2, column=3, padx=5)
        
        Label(self.annotation_frame, text="Total:").grid(row=3, column=0, sticky=W)
        self.total_entry = Entry(self.annotation_frame, width=40)
        self.total_entry.grid(row=3, column=1, pady=5)
        self.total_button = Button(self.annotation_frame, text="Copy", command=lambda: self.copy_to_entry(self.total_entry))
        self.total_button.grid(row=3, column=2, padx=5)
        self.total_color_button = Button(self.annotation_frame, text="Color", command=lambda: self.choose_color("total"))
        self.total_color_button.grid(row=3, column=3, padx=5)
        
        self.save_button = Button(self.annotation_frame, text="Save Annotation", command=self.save_annotation)
        self.save_button.grid(row=4, columnspan=4, pady=10)
        
        self.prev_button = Button(self.annotation_frame, text="Previous Image", command=self.prev_image)
        self.prev_button.grid(row=5, column=0, pady=10)
        
        self.next_button = Button(self.annotation_frame, text="Next Image", command=self.next_image)
        self.next_button.grid(row=5, column=1, pady=10)
        
        self.clear_button = Button(self.annotation_frame, text="Clear Fields", command=self.clear_fields)
        self.clear_button.grid(row=5, column=2, pady=10)

        self.update_ocr_button = Button(self.annotation_frame, text="Update OCR Display", command=self.update_ocr_text_display)
        self.update_ocr_button.grid(row=6, columnspan=4, pady=10)

        self.save_ocr_button = Button(self.annotation_frame, text="Save Updated OCR", command=self.save_ocr_data)
        self.save_ocr_button.grid(row=7, columnspan=4, pady=10)
        
        self.combine_boxes_button = Button(self.annotation_frame, text="Combine Boxes", command=self.combine_boxes)
        self.combine_boxes_button.grid(row=8, columnspan=4, pady=10)

        self.update_box_button = Button(self.annotation_frame, text="Update Box OCR", command=self.update_box_ocr)
        self.update_box_button.grid(row=9, columnspan=4, pady=10)

        self.delete_ocr_button = Button(self.annotation_frame, text="Delete OCR Line", command=self.delete_ocr_line)
        self.delete_ocr_button.grid(row=10, columnspan=4, pady=10)

        self.translate_button = Button(self.annotation_frame, text="Show Translation", command=self.show_translation)
        self.translate_button.grid(row=11, columnspan=4, pady=10)

        self.default_box_color_button = Button(self.annotation_frame, text="Default Box Color", command=self.choose_default_box_color)
        self.default_box_color_button.grid(row=12, columnspan=4, pady=10)
        
        self.load_image()

    def load_unusable_files(self):
        if os.path.exists(self.unusable_files_path):
            with open(self.unusable_files_path, "r") as f:
                return f.read().splitlines()
        return []

    def save_unusable_files(self):
        with open(self.unusable_files_path, "w") as f:
            f.write("\n".join(self.unusable_files))

    def track_unusable_file(self):
        current_image_file = self.image_files[self.current_image_index]
        if current_image_file not in self.unusable_files:
            self.unusable_files.append(current_image_file)
            self.save_unusable_files()
            messagebox.showinfo("Info", f"File {current_image_file} added to unusable files list.")
            self.next_image()

    def update_image_list(self):
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        if self.exclude_annotated.get():
            self.image_files = [f for f in self.image_files if not os.path.exists(os.path.join(self.save_dir, os.path.splitext(f)[0] + ".txt"))]
        if self.exclude_unusable.get():
            self.image_files = [f for f in self.image_files if f not in self.unusable_files]

    def load_image(self):
        if self.current_image_index >= len(self.image_files):
            return

        image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
        self.ocr_path = os.path.join(self.ocr_dir, os.path.splitext(self.image_files[self.current_image_index])[0] + ".txt")
        self.annotation_path = os.path.join(self.save_dir, os.path.splitext(self.image_files[self.current_image_index])[0] + ".txt")

        print("Loading image:", image_path)
        print("Loading OCR data:", self.ocr_path)
        print("Loading annotation data:", self.annotation_path)

        self.image = Image.open(image_path)

        # Apply EXIF orientation
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(self.image._getexif().items())

            if exif[orientation] == 3:
                self.image = self.image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                self.image = self.image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                self.image = self.image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass

        orig_width, orig_height = self.image.size

        max_width = 800
        max_height = 800
        self.image.thumbnail((max_width, max_height), Image.LANCZOS)

        self.scale_x = self.image.width / orig_width
        self.scale_y = self.image.height / orig_height

        self.draw = ImageDraw.Draw(self.image)

        if os.path.exists(self.ocr_path):
            encodings = ["utf-8", "utf-16-le",  "latin1"]  # Add more encodings if needed
            for encoding in encodings:
                try:
                    with open(self.ocr_path, "r", encoding=encoding) as f:
                        self.ocr_data = f.readlines()
                        break  # If reading is successful, break out of the loop
                except UnicodeError:
                    print(f"Failed to read OCR file with encoding: {encoding}")
                    continue
                except Exception as e:
                    messagebox.showerror("Error", f"Error reading OCR file: {self.ocr_path}\n{str(e)}")
                    return

            if not self.ocr_data:
                messagebox.showwarning("Warning", f"No OCR data found in file: {self.ocr_path}")
                self.ocr_listbox.delete(0, END)
                return

            self.ocr_listbox.delete(0, END)
            for line in self.ocr_data:
                parts = line.strip().split(",", 8)
                if len(parts) < 9:
                    print("Skipping invalid OCR line:", line)
                    continue  # Skip invalid lines
                try:
                    coords = list(map(int, parts[:8]))
                except ValueError:
                    print("Skipping line with invalid coordinates:", line)
                    continue  # Skip lines with invalid coordinates
                text = parts[8]

                scaled_coords = [int(coord * self.scale_x if i % 2 == 0 else coord * self.scale_y) for i, coord in enumerate(coords)]

                self.draw.polygon(scaled_coords, outline=self.default_box_color)
                self.draw.text((scaled_coords[0], scaled_coords[1] - 10), text, fill="blue")
                self.ocr_listbox.insert(END, text)
        else:
            messagebox.showwarning("Warning", f"OCR data file not found for image: {image_path}")
            self.ocr_listbox.delete(0, END)

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.tk_image)
        self.clear_fields()
        self.load_annotation()




    def load_annotation(self):
        if os.path.exists(self.annotation_path):
            try:
                with open(self.annotation_path, "r", encoding="utf-8") as f:
                    annotation = json.load(f)
                    print("Annotation loaded:", annotation)  # Debug print
                    self.company_entry.insert(0, annotation.get("company", ""))
                    self.date_entry.insert(0, annotation.get("date", ""))
                    self.address_entry.insert(0, annotation.get("address", ""))
                    self.total_entry.insert(0, annotation.get("total", ""))
                self.refresh_image_with_annotation(annotation)
            except Exception as e:
                messagebox.showerror("Error", f"Error reading annotation file: {self.annotation_path}\n{str(e)}")
        else:
            print("Annotation file not found:", self.annotation_path)  # Debug print

    def save_annotation(self):
        annotation = {
            "company": self.company_entry.get(),
            "date": self.date_entry.get(),
            "address": self.address_entry.get(),
            "total": self.total_entry.get()
        }
        
        annotation_file = os.path.join(self.save_dir, os.path.splitext(self.image_files[self.current_image_index])[0] + ".txt")
        
        with open(annotation_file, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=4, ensure_ascii=False)
        
        messagebox.showinfo("Info", "Annotation saved successfully.")
        self.next_image()

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image()

    def clear_fields(self):
        self.company_entry.delete(0, END)
        self.date_entry.delete(0, END)
        self.address_entry.delete(0, END)
        self.total_entry.delete(0, END)

    def update_ocr_text_display(self):
        try:
            selected_indices = self.ocr_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "No text selected for updating.")
                return
            
            selected_texts = [self.ocr_listbox.get(i) for i in selected_indices]
            new_text = simpledialog.askstring("Update OCR Text", "Enter new OCR text:", initialvalue=" ".join(selected_texts))
            if new_text:
                for i in selected_indices:
                    self.ocr_listbox.delete(i)
                    self.ocr_listbox.insert(i, new_text)

                selected_coords = self.get_selected_coords(selected_indices)
                self.update_ocr_data(selected_coords, new_text)
                self.refresh_image()
        except TclError:
            messagebox.showwarning("Warning", "Error updating OCR text.")

    def combine_boxes(self):
        try:
            selected_indices = self.ocr_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "No text selected for combining.")
                return
            
            selected_texts = [self.ocr_listbox.get(i) for i in selected_indices]
            new_text = simpledialog.askstring("Combine OCR Text", "Enter combined OCR text:", initialvalue=" ".join(selected_texts))
            if new_text:
                for i in reversed(selected_indices):
                    self.ocr_listbox.delete(i)
                self.ocr_listbox.insert(selected_indices[0], new_text)
                
                selected_coords = self.get_selected_coords(selected_indices)
                self.combine_ocr_data(selected_coords, new_text)
                self.save_ocr_data()
                self.refresh_image()
        except TclError:
            messagebox.showwarning("Warning", "Error combining OCR text.")

    def get_selected_coords(self, selected_indices):
        selected_coords = []
        for i in selected_indices:
            parts = self.ocr_data[i].strip().split(",", 8)
            coords = list(map(int, parts[:8]))
            selected_coords.append(coords)
        return selected_coords

    def update_ocr_data(self, selected_coords, new_text):
        for i, line in enumerate(self.ocr_data):
            parts = line.strip().split(",", 8)
            coords = list(map(int, parts[:8]))
            if coords in selected_coords:
                self.ocr_data[i] = ",".join(map(str, coords)) + "," + new_text + "\n"
                break

    def combine_ocr_data(self, selected_coords, new_text):
        min_x = min([coord[0] for coord in selected_coords])
        min_y = min([coord[1] for coord in selected_coords])
        max_x = max([coord[4] for coord in selected_coords])
        max_y = max([coord[5] for coord in selected_coords])
        combined_coords = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        
        new_ocr_data = ",".join(map(str, combined_coords)) + "," + new_text + "\n"
        self.ocr_data = [line for line in self.ocr_data if list(map(int, line.strip().split(",", 8)[:8])) not in selected_coords]
        self.ocr_data.append(new_ocr_data)

    def save_ocr_data(self):
        with open(self.ocr_path, "w", encoding="utf-16") as f:
            f.writelines(self.ocr_data)
        messagebox.showinfo("Info", "OCR data saved successfully.")

    def update_box_ocr(self):
        try:
            selected_indices = self.ocr_listbox.curselection()
            if not selected_indices or len(selected_indices) != 1:
                messagebox.showwarning("Warning", "Select exactly one OCR line to update.")
                return
            
            selected_index = selected_indices[0]
            selected_text = self.ocr_listbox.get(selected_index)
            new_text = simpledialog.askstring("Update Box OCR", "Enter new OCR text:", initialvalue=selected_text)
            if new_text:
                self.ocr_listbox.delete(selected_index)
                self.ocr_listbox.insert(selected_index, new_text)
                
                coords = self.get_selected_coords([selected_index])[0]
                self.update_ocr_data([coords], new_text)
                self.save_ocr_data()
                self.refresh_image()
        except TclError:
            messagebox.showwarning("Warning", "Error updating OCR text.")

    def delete_ocr_line(self):
        try:
            selected_indices = self.ocr_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "No text selected for deletion.")
                return
            
            for i in reversed(selected_indices):
                self.ocr_listbox.delete(i)
            
            self.ocr_data = [line for idx, line in enumerate(self.ocr_data) if idx not in selected_indices]
            self.save_ocr_data()
            self.refresh_image()
        except TclError:
            messagebox.showwarning("Warning", "Error deleting OCR line.")

    def refresh_image(self):
        self.image = Image.open(os.path.join(self.image_dir, self.image_files[self.current_image_index]))
        
        # Apply EXIF orientation
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = dict(self.image._getexif().items())

            if exif[orientation] == 3:
                self.image = self.image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                self.image = self.image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                self.image = self.image.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            pass
        
        orig_width, orig_height = self.image.size
        max_width = 800
        max_height = 1000
        self.image.thumbnail((max_width, max_height), Image.LANCZOS)
        
        self.scale_x = self.image.width / orig_width
        self.scale_y = self.image.height / orig_height
        
        self.draw = ImageDraw.Draw(self.image)
        
        for line in self.ocr_data:
            parts = line.strip().split(",", 8)
            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                continue  # Skip this line if coordinates are not valid integers
            text = parts[8]
            scaled_coords = [int(coord * self.scale_x if i % 2 == 0 else coord * self.scale_y) for i, coord in enumerate(coords)]
            self.draw.polygon(scaled_coords, outline=self.default_box_color)
            self.draw.text((scaled_coords[0], scaled_coords[1] - 10), text, fill="blue")
        
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.tk_image)

    def refresh_image_with_annotation(self, annotation):
        self.refresh_image()
        for key, color in self.class_colors.items():
            if annotation.get(key):
                for line in self.ocr_data:
                    parts = line.strip().split(",", 8)
                    try:
                        coords = list(map(int, parts[:8]))
                    except ValueError:
                        continue  # Skip this line if coordinates are not valid integers
                    text = parts[8]
                    if annotation[key].strip() in text.strip():
                        scaled_coords = [int(coord * self.scale_x if i % 2 == 0 else coord * self.scale_y) for i, coord in enumerate(coords)]
                        self.draw.polygon(scaled_coords, outline=color)
                        self.draw.text((scaled_coords[0], scaled_coords[1] - 10), text, fill=color)

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.tk_image)

    def choose_color(self, class_name):
        color = colorchooser.askcolor(title=f"Choose color for {class_name}")[1]
        if color:
            self.class_colors[class_name] = color

    def choose_default_box_color(self):
        color = colorchooser.askcolor(title="Choose default box color")[1]
        if color:
            self.default_box_color = color

    def copy_to_entry(self, entry):
        try:
            selected_indices = self.ocr_listbox.curselection()
            if not selected_indices:
                messagebox.showwarning("Warning", "No text selected for copying.")
                return
            
            selected_texts = [self.ocr_listbox.get(i) for i in selected_indices]
            entry.delete(0, END)
            entry.insert(0, " ".join(selected_texts))
        except TclError:
            messagebox.showwarning("Warning", "Error copying text.")

    def show_translation(self):
        translation_window = Toplevel(self.master)
        translation_window.title("Translated Text")
        translation_window.geometry("800x400")

        text_frame = Frame(translation_window)
        text_frame.pack(expand=1, fill=BOTH, padx=10, pady=10)

        scrollbar = Scrollbar(text_frame)
        scrollbar.pack(side=RIGHT, fill=Y)

        original_text_label = Listbox(text_frame, selectmode=SINGLE, yscrollcommand=scrollbar.set, width=50)
        original_text_label.pack(side=LEFT, expand=1, fill=BOTH, padx=5)

        translated_text_label = Text(text_frame, wrap=WORD, yscrollcommand=scrollbar.set, width=50)
        translated_text_label.pack(side=LEFT, expand=1, fill=BOTH, padx=5)

        scrollbar.config(command=self.sync_scrollbars)

        self.original_text_label = original_text_label
        self.translated_text_label = translated_text_label

        # Bind mouse scroll to both text widgets
        original_text_label.bind("<MouseWheel>", self.sync_scroll)
        translated_text_label.bind("<MouseWheel>", self.sync_scroll)

        # Fetch the translated text using googletrans
        try:
            translator = Translator()
            original_text = "\n".join(self.ocr_listbox.get(0, END))
            translation = translator.translate(original_text, dest='en')
            for line in original_text.split("\n"):
                original_text_label.insert(END, line)
            translated_text_label.insert(END, translation.text)
        except Exception as e:
            messagebox.showerror("Error", f"Error fetching translation: {str(e)}")

        highlight_button = Button(translation_window, text="Highlight", command=self.highlight_selection)
        highlight_button.pack(side=LEFT, padx=10, pady=10)

        reset_button = Button(translation_window, text="Reset View", command=self.reset_view)
        reset_button.pack(side=LEFT, padx=10, pady=10)

    def sync_scrollbars(self, *args):
        self.original_text_label.yview(*args)
        self.translated_text_label.yview(*args)

    def sync_scroll(self, event):
        self.original_text_label.yview_scroll(int(-1*(event.delta/120)), "units")
        self.translated_text_label.yview_scroll(int(-1*(event.delta/120)), "units")
        return "break"

    def highlight_selection(self):
        try:
            selected_index = self.original_text_label.curselection()
            if not selected_index:
                messagebox.showwarning("Warning", "No text selected for highlighting.")
                return

            selected_index = selected_index[0]
            self.refresh_image()
            coords = self.get_selected_coords([selected_index])[0]
            scaled_coords = [int(coord * self.scale_x if i % 2 == 0 else coord * self.scale_y) for i, coord in enumerate(coords)]
            top_left = (scaled_coords[0], scaled_coords[1])
            bottom_right = (scaled_coords[4], scaled_coords[5])
            self.draw.rectangle([top_left, bottom_right], outline="yellow", width=3)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.image_label.config(image=self.tk_image)
        except TclError:
            messagebox.showwarning("Warning", "Error highlighting selection.")

    def reset_view(self):
        self.refresh_image()

if __name__ == "__main__":
    root = Tk()
    app = Annotator(root)
    root.mainloop()
