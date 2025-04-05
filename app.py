from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os
import requests
import gdown  # <-- NEW

app = Flask(__name__)
CORS(app, resources=r'/*', origins='*')

# Constants
FILE_ID = "12AHcBgU5nfZ2LFATLZJJaUf0eJNN6ZX_"
MODEL_PATH = "trained_model.keras"

# Use gdown to download model correctly
def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive using gdown...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")


# Download and load model
download_model_if_needed()
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class names
class_name = ['24 Avatars', '84 Pillared Cenotaph', 'Achalgarh Fort', 'Adhai Din Ka Jhonpra', 
    'Adi Badri Temple', 'Agnitheertham', 'Agra Fort', 'Aina Mahal', 'Ajmer Sharif Dargah', 
    'Akshardham Temple', 'Alakhnath Temple', 'Albert Hall Museum', 'Allahabad Fort', 'Anand Bhawan', 
    'Anandpur Sahib Gurudwara', 'Ananthapura Lake Temple', 'Arthuna Temples', 'Badaun Fort', 
    'Badrinath Temple', 'Bagnath Temple', 'Bahadurgarh Fort', 'Bala Quila (Alwar Fort)', 
    'Balancing Rocks', 'Bara Imambara', 'Bara Kaman', 'Beypore', 'Bhadra Fort', 'Bhalka Tirtha', 
    'Bhangarh Fort (haunted fort)', 'Birla Mandir', 'Birla Temple', 'Brahma Temple', 
    'Brihadeeswarar Temple', 'Buddha Park (Tathagata Tsal)', 'Buddha Vihar', 'Chandi Devi Temple', 
    'Char Dham', 'Chittorgarh Fort', 'Chota Imambara', 'City Palace', 'Dargah-e-Alahazrat', 
    'Deeg Palace', 'Devi Talab Mandir', 'Dilwara Temples', 'Dubdi Monastery', 'Durgiana Temple', 
    'Dwarkadhish Temple', 'Eachanari Vinayagar Temple', 'Ekambareswarar Temple', 'Eklingji Temple', 
    'Enchey Monastery', 'Fort St. George', 'Gandhi hall', 'Gangotri Temple', 'Garadia Mahadev Temple', 
    'Garh Palace', 'Girnar Hill and Temple', 'Gita Press', 'Gobindgarh Fort', 'Gohar Mahal', 
    'Gol Gumbaz', 'Golden Temple', 'Gopi Talav', 'Gorakhnath Temple', 'Goravanahalli Lakshmi Temple', 
    "Governor's House (Raj Bhavan)", 'Great Rann of Kutch', 'Guru Shikhar', 'Gurudwara nada Sahib', 
    'Guruvayur Temple', 'Gwalior Fort', 'Hampi Group of Monuments', 'Hanuman Tok', 'Har Ki Pauri', 
    'Hawa Mahal', 'Hemkund Sahib', 'Hoshangs Tomb', 'Ibrahim Rauza', 'Imam Nasir Mausoleum', 
    "Itimad-ud-Daulah's Tomb", 'JK Temple', 'Jag Mandir', 'Jagatjit Palace', 'Jagdish Temple', 
    'Jageshwar Dham', 'Jaigarh Fort', 'Jaisalmer Fort', 'Jalakandeswarar Temple', 'Jallianwala Bagh', 
    'Jami Masjid', 'Jantar Mantar', 'Jaswant Thada', "Jatayu Earth's Center", 'Javari Temple', 
    'Jewish Synagogue', 'Jhansi Fort', 'Junagarh Fort', 'Jwalpa Devi Temple', 'Kaba Gandhi No Delo', 
    'Kadri Manjunath Temple', 'Kailasanathar Temple', 'Kalpeshwar Temple', 'Kanakakunnu Palace', 
    'Kandoliya Temple', 'Kanpur Memorial Church', 'Kapaleeshwarar Temple', 'Kapileshwar Mahadev Temple', 
    'Karni Mata Temple (Rat Temple)', 'Kasar Devi Temple', 'Kashi Vishwanath Temple', 
    'Kedarnath Temple', 'Khwaja Bande Nawaz Dargah', 'Kirti Mandir', 'Kolar Gold Fields', 
    'Kota Barrage', 'Krishna Janmabhoomi', 'Krishna Temple', 'Kushinagar', 'Lachen Monastery', 
    'Lachung Monastery', 'Lakkundi', 'Lal Bagh Palace', 'Lalgarh Palace', 'Laxman Jhula and Ram Jhula', 
    'Laxmi Vilas Palace', 'Lohagarh Fort', 'Madan mahal', 'Madikeri Fort', 'Mahabat Maqbara', 
    'Maharaja Ranjit Singh War Museum', 'Maharao Madho Singh Museum', 'Mahatma Mandir', 
    'Mahudi Jain Temple', 'Mandore Gardens', 'Mansa Devi Temple', 'Marudhamalai Temple', 
    'Mazaar of Peer Haji Rattan', 'Meenakshi Amman Temple', 'Meera Temple', 'Mehrangarh Fort', 
    'Mindrolling Monastery', 'Mirjan Fort', 'Mishkal Mosque', 'Moorish Mosque', 
    'Moosi Maharani ki Chhatri', 'Moti Bagh Palace', 'Mysore Palace', 'Nageshwar Jyotirlinga Temple', 
    'Nahargarh Fort', 'Naina Devi Temple', 'Namchi Monastery', 'Nanda Devi Temple', 'Nareli Jain Temple', 
    'Neelkanth Mahadev Temple', 'Neyyar Dam', 'Nilambag Palace', 'Nilgiri Mountain Railway', 
    'Nishkalank Mahadev Temple', 'Norbugang Coronation Throne', 'Ooty Botanical Gardens', 
    'Orchha Fort', 'Our Lady of Ransom Church', 'Padmanabhaswamy Temple', 
    'Patal Bhuvaneshwar Cave Temple', 'Patwon ki Haveli', 'Pelling', 'Perur Pateeswarar Temple', 
    'Phensang Monastery', 'Phodong Monastery', 'Ponmudi', 'Prabhas Patan Museum', 'Prag Mahal', 
    'Punjab Agricultural University Museum', 'Pushkar Camel Fair (annual event)', 'Pushkar Lake', 
    'Qila Anandgarh Sahib', 'Qila Mubarak Bathinda', 'Rajwada Palace', 'Ralang Monastery', 
    'Ramanathaswamy Temple', 'Rampuria Havelis', 'Rani Mahal', 'Rani Padmini’s Palace', 
    'Ranthambore Fort', 'Rotary Dolls Museum', 'Royal Chhatris', 'Rukmini Devi Temple', 
    'Rumi Darwaza', 'Rumtek Monastery', 'Sabarmati Ashram', 'Sajjangarh Palace (Monsoon Palace)', 
    'Salim Singh ki Haveli', 'Samdruptse Monastery', 'Sanga Choeling Monastery', 
    'Santa Cruz Basilica', 'Santhebennur Pushkarani', 'Santhome Basilica', 'Sarnath', 
    'Savitri Temple', 'Science Centre', 'Shahid Smarak', 'Shakthan Thampuran Palace', 
    'Shakti Temple', 'Shantinath temple', 'Sheesh Mahal', 'Shree Mahakaleshwar Temple', 
    'Shri Bankey Bihari Mandir', 'Shri Ram Raja Mandir', 'Sidi Saiyyed Mosque', 
    'Singhik Viewpoint and Temple', 'Solophok Char Dham', 'Someshwara Temple', 'Somnath Temple', 
    'Sri Ranganathaswamy Temple, Srirangam', 'Srirangapatna', 'St Marys Forane Church', 
    'St. Andrews Basilica', "St. Angelo's Fort", 'St. John in the Wilderness Church', 
    'St. Lourdes Church', "St. Mary's English Church", 'St. Marys Cathedral Church', 
    "St. Philomena's Church", 'Sukh Mahal', 'Sun Temple', 'Taj Mahal', 'Taj-ul-Masajid', 
    'Takht Sri Damdama Sahib', 'Takhteshwar Temple', 'Tansen memorial', 'Tapkeshwar Temple', 
    'Taragarh Fort', 'Tea Museum', 'Tendong Hill', 'Thambi Viewpoint and Kali Mandir', 
    'Thanjavur Palace', 'Thirumalai Nayakkar Mahal', 'Thiruvalluvar Statue', 'Tilwara Ghat', 
    'Tour Krishnapuram Palace', 'Trikuteshwara Temple', 'Trinetra Ganesh Temple', 
    'Trivati Nath Temple', 'Triveni Ghat', 'Triveni Sangam Temple', 'Triyuginarayan Temple', 
    'Tsomgo Lake and Baba Mandir', 'Umaid Bhawan Palace', 'Uparkot Fort', 
    'Vijay Stambh (Victory Tower)', 'Vijaya Vittala Temple', 'Virasat-e-Khalsa Museum', 
    'Virupaksha Temple', 'Watson Museum', 'Yamunotri Temple', 'Yana Rocks', 'amber fort', 
    'ashrafi mahal', 'kandariya mahadeva temple', 'khajrana ganesh mandir', 
    'pazhassi raja archaeological museum', 'siddhanath temple']

# Prediction function
def predict_image(image):
    try:
        image = image.resize((64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        return result_index
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# Routes
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Backend is running"}), 200

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    try:
        image = Image.open(file.stream)
        result_index = predict_image(image)
        if result_index is None:
            return jsonify({"error": "Prediction failed"}), 500

        predicted_class_name = class_name[result_index]
        return jsonify({"predicted_class": predicted_class_name})

    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
