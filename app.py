import os
import string
import requests
import torch
import markdown
from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification

# LangChain imports
from langchain_core.language_models import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, ChatGeneration

# -----------------------
# Load environment variables
load_dotenv()

# -----------------------
# Flask setup
app = Flask(__name__)

# -----------------------
# MongoDB for disease info
mongodb_password = os.getenv("MONGODB_PASSWORD")
client = MongoClient(
    f"mongodb+srv://samratray:{mongodb_password}"
    "@cluster0.fcgztso.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
db = client["medichat"]
collection = db["disease_info"]

def retrieve_info_from_mongo(disease_name):
    doc = collection.find_one(
        {"Disease": {"$regex": f"^{disease_name}$", "$options": "i"}}
    )
    if doc:
        return {"disease": disease_name, "description": doc["Description"]}
    return {"disease": None, "description": "Information not found for the given disease."}

# -----------------------
# Load BERT models
# 1) Symptom classifier
initial_bert_dir = "./models/my_bert_model"
initial_bert_model = BertForSequenceClassification.from_pretrained(initial_bert_dir)
initial_bert_tokenizer = BertTokenizer.from_pretrained(initial_bert_dir)

# 2) Disease predictor
bert_model_dir = "./models/my_bert_model2"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_dir)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

initial_bert_model.to(device)
bert_model.to(device)


# ----- Initialize from Text Files -----
# Populate your disease_names and symptoms as needed.
disease_names = ['panic disorder', 'vocal cord polyp', 'turner syndrome', 'cryptorchidism', 'poisoning due to ethylene glycol', 'atrophic vaginitis', 'fracture of the hand', 'cellulitis or abscess of mouth', 'eye alignment disorder', 'headache after lumbar puncture', 'pyloric stenosis', 'salivary gland disorder', 'osteochondrosis', 'injury to the knee', 'metabolic disorder', 'vaginitis', 'sick sinus syndrome', 'tinnitus of unknown cause', 'glaucoma', 'eating disorder', 'transient ischemic attack', 'pyelonephritis', 'rotator cuff injury', 'chronic pain disorder', 'problem during pregnancy', 'liver cancer', 'atelectasis', 'injury to the hand', 'choledocholithiasis', 'injury to the hip', 'cirrhosis', 'thoracic aortic aneurysm', 'subdural hemorrhage', 'diabetic retinopathy', 'fibromyalgia', 'ischemia of the bowel', 'fetal alcohol syndrome', 'peritonitis', 'injury to the abdomen', 'acute pancreatitis', 'thrombophlebitis', 'asthma', 'foreign body in the vagina', 'restless leg syndrome', 'emphysema', 'cysticercosis', 'induced abortion', 'teething syndrome', 'infectious gastroenteritis', 'acute sinusitis', 'substance-related mental disorder', 'postpartum depression', 'coronary atherosclerosis', 'spondylitis', 'pituitary adenoma', 'uterine fibroids', 'idiopathic nonmenstrual bleeding', 'chalazion', 'ovarian torsion', 'retinopathy due to high blood pressure', 'vaginal yeast infection', 'mastoiditis', 'lung contusion', 'hypertrophic obstructive cardiomyopathy (hocm)', 'ingrown toe nail', 'pulmonary eosinophilia', 'corneal disorder', 'foreign body in the gastrointestinal tract', 'endophthalmitis', 'intestinal malabsorption', 'viral warts', 'hyperhidrosis', 'stroke', 'pilonidal cyst', 'crushing injury', 'normal pressure hydrocephalus', 'alopecia', 'hashimoto thyroiditis', 'flat feet', 'nonalcoholic liver disease (nash)', 'hemarthrosis', 'pelvic organ prolapse', 'fracture of the arm', 'coagulation (bleeding) disorder', 'intracranial hemorrhage', 'hyperkalemia', 'cornea infection', 'abscess of the lung', 'dengue fever', 'chronic sinusitis', 'cholesteatoma', 'volvulus', 'injury to the finger', 'poisoning due to analgesics', 'atrial fibrillation', 'pinworm infection', 'urethral valves', 'open wound of the neck', 'achalasia', 'conductive hearing loss', 'abdominal hernia', 'cerebral palsy', 'marijuana abuse', 'cryptococcosis', 'obesity', 'indigestion', 'bursitis', 'esophageal cancer', 'pulmonary congestion', 'juvenile rheumatoid arthritis', 'actinic keratosis', 'acute otitis media', 'astigmatism', 'tuberous sclerosis', 'empyema', 'presbyacusis', 'neonatal jaundice', 'chronic obstructive pulmonary disease (copd)', 'dislocation of the elbow', 'spondylosis', 'herpangina', 'injury to the shoulder', 'poisoning due to antidepressants', 'infection of open wound', 'deep vein thrombosis (dvt)', 'protein deficiency', 'myoclonus', 'bone spur of the calcaneous', 'von willebrand disease', 'open wound of the back', 'heart block', 'colonic polyp', 'magnesium deficiency', 'female infertility of unknown cause', 'pericarditis', 'attention deficit hyperactivity disorder (adhd)', 'pulmonic valve disease', 'tietze syndrome', 'cranial nerve palsy', 'injury to the arm', 'conversion disorder', 'complex regional pain syndrome', 'otosclerosis', 'injury to the trunk', 'hypothyroidism', 'primary insomnia', 'lice', 'vitamin b12 deficiency', 'diabetes', 'vulvodynia', 'endometriosis', 'vasculitis', 'concussion', 'oral leukoplakia', 'chronic kidney disease', 'bladder disorder', 'chorioretinitis', 'priapism', 'myositis', 'mononucleosis', 'neuralgia', 'polycystic kidney disease', 'bipolar disorder', 'amyloidosis', 'chronic inflammatory demyelinating polyneuropathy (cidp)', 'gastroesophageal reflux disease (gerd)', 'vitreous hemorrhage', 'poisoning due to antimicrobial drugs', 'open wound of the mouth', 'scleroderma', 'myasthenia gravis', 'hypoglycemia', 'idiopathic absence of menstruation', 'dislocation of the ankle', 'carbon monoxide poisoning', 'panic attack', 'plantar fasciitis', 'hyperopia', 'poisoning due to sedatives', 'pemphigus', 'peyronie disease', 'hiatal hernia', 'extrapyramidal effect of drugs', 'meniere disease', 'anal fissure', 'allergy', 'chronic otitis media', 'fracture of the finger', 'hirschsprung disease', 'polymyalgia rheumatica', 'lymphedema', 'bladder cancer', 'acute bronchospasm', 'acute glaucoma', 'open wound of the chest', 'dislocation of the patella', 'sciatica', 'hypercalcemia', 'stress incontinence', 'varicose veins', 'benign kidney cyst', 'hydrocele of the testicle', 'degenerative disc disease', 'hirsutism', 'dislocation of the foot', 'hydronephrosis', 'diverticulosis', 'pain after an operation', 'huntington disease', 'lymphoma', 'dermatitis due to sun exposure', 'anemia due to chronic kidney disease', 'injury to internal organ', 'scleritis', 'pterygium', 'fungal infection of the skin', 'insulin overdose', 'syndrome of inappropriate secretion of adh (siadh)', 'foreign body in the ear', 'premenstrual tension syndrome', 'orbital cellulitis', 'injury to the leg', 'hepatic encephalopathy', 'bone cancer', 'syringomyelia', 'pulmonary fibrosis', 'mitral valve disease', 'parkinson disease', 'gout', 'otitis media', 'drug abuse (opioids)', 'myelodysplastic syndrome', 'fracture of the shoulder', 'acute kidney injury', 'threatened pregnancy', 'intracranial abscess', 'gum disease', 'open wound from surgical incision', 'gastrointestinal hemorrhage', 'seborrheic dermatitis', 'drug abuse (methamphetamine)', 'torticollis', 'poisoning due to antihypertensives', 'tension headache', 'alcohol intoxication', 'scurvy', 'narcolepsy', 'food allergy', 'labyrinthitis', 'anxiety', 'impulse control disorder', 'stenosis of the tear duct', 'abscess of nose', 'omphalitis', 'leukemia', 'bell palsy', 'conjunctivitis due to allergy', 'drug reaction', 'adrenal cancer', 'myopia', 'osteoarthritis', 'thyroid disease', 'pharyngitis', 'chronic rheumatic fever', 'hypocalcemia', 'macular degeneration', 'pneumonia', 'cold sore', 'premature ventricular contractions (pvcs)', 'testicular cancer', 'hydrocephalus', 'breast cancer', 'anemia due to malignancy', 'esophageal varices', 'endometrial cancer', 'cystic fibrosis', 'intertrigo (skin condition)', 'parathyroid adenoma', 'glucocorticoid deficiency', 'temporomandibular joint disorder', 'wilson disease', 'vesicoureteral reflux', 'vitamin a deficiency', 'gonorrhea', 'fracture of the rib', 'ependymoma', 'hepatitis due to a toxin', 'vaginal cyst', 'open wound of the shoulder', 'ectopic pregnancy', 'chronic knee pain', 'pinguecula', 'hypergammaglobulinemia', 'alcohol abuse', 'carpal tunnel syndrome', 'pituitary disorder', 'kidney stone', 'autism', 'cat scratch disease', 'chronic glaucoma', 'retinal detachment', 'aplastic anemia', 'overflow incontinence', 'hemolytic anemia', 'lateral epicondylitis (tennis elbow)', 'open wound of the eye', 'syphilis', 'diabetic kidney disease', 'nose disorder', 'drug withdrawal', 'dental caries', 'hypercholesterolemia', 'fracture of the patella', 'kidney failure', 'fracture of the neck', 'muscle spasm', 'hemophilia', 'hyperosmotic hyperketotic state', 'peritonsillar abscess', 'gastroparesis', 'itching of unknown cause', 'polycythemia vera', 'thrombocytopenia', 'head and neck cancer', 'pseudohypoparathyroidism', 'goiter', 'urge incontinence', 'edward syndrome', 'open wound of the arm', 'muscular dystrophy', 'mittelschmerz', 'corneal abrasion', 'anemia of chronic disease', 'dysthymic disorder', 'scarlet fever', 'hypertensive heart disease', 'drug abuse (barbiturates)', 'polycystic ovarian syndrome (pcos)', 'encephalitis', 'cyst of the eyelid', 'balanitis', 'foreign body in the throat', 'drug abuse (cocaine)', 'optic neuritis', 'alcohol withdrawal', 'premature atrial contractions (pacs)', 'hemiplegia', 'hammer toe', 'open wound of the cheek', 'joint effusion', 'open wound of the knee', 'meningioma', 'brain cancer', 'placental abruption', 'seasonal allergies (hay fever)', 'lung cancer', 'primary kidney disease', 'uterine cancer', 'dry eye of unknown cause', 'fibrocystic breast disease', 'fungal infection of the hair', 'tooth abscess', 'envenomation from spider or animal bite', 'vacterl syndrome', 'vertebrobasilar insufficiency', 'rectal disorder', 'atonic bladder', 'benign paroxysmal positional vertical (bppv)', 'blepharospasm', 'sarcoidosis', 'metastatic cancer', 'trigger finger (finger disorder)', 'stye', 'hemochromatosis', 'osteochondroma', 'cushing syndrome', 'typhoid fever', 'vitreous degeneration', 'atrophic skin condition', 'aspergillosis', 'uterine atony', 'trichinosis', 'whooping cough', 'open wound of the lip', 'subacute thyroiditis', 'oral mucosal lesion', 'open wound due to trauma', 'intracerebral hemorrhage', 'alzheimer disease', 'vaginismus', 'systemic lupus erythematosis (sle)', 'premature ovarian failure', 'thoracic outlet syndrome', 'ganglion cyst', 'dislocation of the knee', 'crohn disease', 'postoperative infection', 'folate deficiency', 'fluid overload', 'atrial flutter', 'skin disorder', 'floaters', 'tooth disorder', 'heart attack', 'open wound of the abdomen', 'fracture of the leg', 'oral thrush (yeast infection)', 'pityriasis rosea', 'allergy to animals', 'orthostatic hypotension', 'obstructive sleep apnea (osa)', 'hypokalemia', 'psoriasis', 'dislocation of the shoulder', 'intussusception', 'cervicitis', 'abscess of the pharynx', 'primary thrombocythemia', 'arthritis of the hip', 'decubitus ulcer', 'hypernatremia', 'sensorineural hearing loss', 'chronic ulcer', 'osteoporosis', 'ileus', 'sickle cell crisis', 'urethritis', 'prostatitis', "otitis externa (swimmer's ear)", 'poisoning due to anticonvulsants', 'testicular torsion', 'tricuspid valve disease', 'urethral stricture', 'vitamin d deficiency', 'hydatidiform mole', 'pain disorder affecting the neck', 'tuberculosis', 'pelvic fistula', 'acute bronchiolitis', 'presbyopia', 'dementia', 'insect bite', 'paroxysmal ventricular tachycardia', 'congenital heart defect', 'connective tissue disorder', 'foreign body in the eye', 'poisoning due to gas', 'pyogenic skin infection', 'endometrial hyperplasia', 'acanthosis nigricans', 'central atherosclerosis', 'viral exanthem', 'noninfectious gastroenteritis', 'benign prostatic hyperplasia (bph)', 'menopause', 'primary immunodeficiency', 'ovarian cancer', 'cataract', 'dislocation of the hip', 'spinal stenosis', 'intestinal obstruction', 'heart contusion', 'congenital malformation syndrome', 'sporotrichosis', 'lymphangitis', 'wernicke korsakoff syndrome', 'intestinal disease', 'acute bronchitis', 'persistent vomiting of unknown cause', 'open wound of the foot', 'myocarditis', 'preeclampsia', 'ischemic heart disease', 'neurofibromatosis', 'chickenpox', 'pancreatic cancer', 'neuropathy due to drugs', 'croup', 'idiopathic excessive menstruation', 'amblyopia', 'meckel diverticulum', 'dislocation of the wrist', 'ear drum damage', 'erectile dysfunction', 'temporary or benign blood in urine', 'kidney disease due to longstanding hypertension', 'chondromalacia of the patella', 'onychomycosis', 'urethral disorder', 'lyme disease', 'iron deficiency anemia', 'acute respiratory distress syndrome (ards)', 'toxic multinodular goiter', 'open wound of the finger', 'autonomic nervous system disorder', 'psychosexual disorder', 'anemia', 'tendinitis', 'common cold', 'amyotrophic lateral sclerosis (als)', 'central retinal artery or vein occlusion', 'paroxysmal supraventricular tachycardia', 'venous insufficiency', 'trichomonas infection', 'acne', 'depression', 'drug abuse', 'urinary tract obstruction', 'diabetes insipidus', 'iridocyclitis', 'varicocele of the testicles', 'irritable bowel syndrome', 'fracture of the foot', 'ovarian cyst', 'chlamydia', 'parasitic disease', 'fracture of the jaw', 'lipoma', 'female genitalia infection', 'pulmonary hypertension', 'thyroid nodule', 'broken tooth', 'dumping syndrome', 'lymphadenitis', 'injury to the face', 'aortic valve disease', 'rheumatoid arthritis', 'spermatocele', 'impetigo', 'anal fistula', 'hypothermia', 'oppositional disorder', 'migraine', 'diabetic peripheral neuropathy', 'testicular disorder', 'gestational diabetes', 'hidradenitis suppurativa', 'valley fever', 'conjunctivitis due to bacteria', 'lewy body dementia', 'multiple myeloma', 'head injury', 'ascending cholangitis', 'idiopathic irregular menstrual cycle', 'interstitial lung disease', 'mononeuritis', 'malaria', 'somatization disorder', 'hypovolemia', 'schizophrenia', 'knee ligament or meniscus tear', 'endocarditis', 'sepsis', 'heat stroke', 'cholecystitis', 'cardiac arrest', 'cardiomyopathy', 'social phobia', 'meningitis', 'spherocytosis', 'hormone disorder', 'raynaud disease', 'reactive arthritis', 'scabies', 'ear wax impaction', 'hypertension of pregnancy', 'peripheral arterial embolism', 'rosacea', 'fracture of the skull', 'uveitis', 'fracture of the facial bones', 'tracheitis', 'jaw disorder', 'perirectal infection', 'breast cyst', 'post-traumatic stress disorder (ptsd)', 'kidney cancer', 'vulvar cancer', 'blepharitis', 'celiac disease', 'cystitis', 'sickle cell anemia', 'subconjunctival hemorrhage', 'hemorrhoids', 'contact dermatitis', 'sinus bradycardia', 'high blood pressure', 'pelvic inflammatory disease', 'liver disease', 'chronic constipation', 'thyroid cancer', 'flu', 'friedrich ataxia', 'tic (movement) disorder', 'skin polyp', 'brachial neuritis', 'cervical cancer', 'adrenal adenoma', 'esophagitis', 'gas gangrene', 'yeast infection', 'spina bifida', 'drug poisoning due to medication', 'alcoholic liver disease', 'malignant hypertension', 'diverticulitis', 'moyamoya disease', 'heat exhaustion', 'psychotic disorder', 'frostbite', 'atrophy of the corpus cavernosum', 'smoking or tobacco addiction', 'sprain or strain', 'essential tremor', 'open wound of the ear', 'foreign body in the nose', 'idiopathic painful menstruation', 'down syndrome', 'idiopathic infrequent menstruation', 'pneumothorax', 'de quervain disease', 'fracture of the vertebra', 'human immunodeficiency virus infection (hiv)', 'mumps', 'subarachnoid hemorrhage', 'acute fatty liver of pregnancy (aflp)', 'ectropion', 'scar', 'lactose intolerance', 'eustachian tube dysfunction (ear disorder)', 'appendicitis', 'graves disease', 'dissociative disorder', 'open wound of the face', 'dislocation of the vertebra', 'phimosis', 'hyperemesis gravidarum', 'pregnancy', 'thalassemia', 'placenta previa', 'epidural hemorrhage', 'septic arthritis', "athlete's foot", 'pleural effusion', 'aphakia', 'vulvar disorder', 'sialoadenitis', 'gynecomastia', 'urinary tract infection', 'histoplasmosis', 'erythema multiforme', 'scoliosis', 'bunion', 'arrhythmia', 'trigeminal neuralgia', 'ankylosing spondylitis', 'peripheral nerve disorder', 'sebaceous cyst', 'poisoning due to antipsychotics', 'neurosis', 'prostate cancer', 'cerebral edema', 'dislocation of the finger', 'birth trauma', 'chronic pancreatitis', 'hematoma', 'carcinoid syndrome', 'open wound of the head', 'seborrheic keratosis', 'burn', 'spontaneous abortion', 'genital herpes', 'adjustment reaction', 'gallstone', 'multiple sclerosis', 'zenker diverticulum', 'fracture of the pelvis', 'pneumoconiosis', 'hyperlipidemia', 'ulcerative colitis', 'male genitalia infection', 'hpv', 'angina', 'injury to the spinal cord', 'nasal polyp', 'lichen simplex', 'trichiasis', 'acariasis', 'colorectal cancer', 'skin pigmentation disorder', 'factitious disorder', 'lymphogranuloma venereum', 'galactorrhea of unknown cause', 'g6pd enzyme deficiency', 'nerve impingement near the shoulder', 'toxoplasmosis', 'fibroadenoma', 'open wound of the hand', 'missed abortion', 'diabetic ketoacidosis', 'granuloma inguinale', 'obsessive compulsive disorder (ocd)', 'injury of the ankle', 'hyponatremia', 'stricture of the esophagus', 'fracture of the ankle', 'soft tissue sarcoma', 'bone disorder', 'epilepsy', 'personality disorder', 'shingles (herpes zoster)', 'tourette syndrome', 'avascular necrosis', 'strep throat', 'spinocerebellar ataxia', 'osteomyelitis', 'sjogren syndrome', 'adhesive capsulitis of the shoulder', 'viral hepatitis', 'tonsillar hypertrophy', 'gastritis', 'skin cancer', 'rheumatic fever', 'aphthous ulcer', 'tonsillitis', 'intestinal cancer', 'rocky mountain spotted fever', 'stomach cancer', 'developmental disability', 'acute stress reaction', 'delirium', 'callus', 'guillain barre syndrome', 'lumbago', 'deviated nasal septum', 'hemangioma', 'peripheral arterial disease', 'chronic back pain', 'heart failure', 'conjunctivitis', 'herniated disk', 'rhabdomyolysis', 'breast infection (mastitis)', 'abdominal aortic aneurysm', 'pulmonary embolism', 'conduct disorder', 'mastectomy', 'epididymitis', 'premature rupture of amniotic membrane', 'molluscum contagiosum', 'necrotizing fasciitis', 'benign vaginal discharge (leukorrhea)', 'bladder obstruction', 'melanoma', 'cervical disorder', 'laryngitis', 'dyshidrosis', 'poisoning due to opioids', 'diaper rash', 'lichen planus', 'gastroduodenal ulcer', 'inguinal hernia', 'eczema', 'asperger syndrome', 'mucositis', 'paronychia', 'open wound of the jaw', 'white blood cell disease', 'kaposi sarcoma', 'spondylolisthesis', 'pseudotumor cerebri', 'conjunctivitis due to virus', 'open wound of the nose']


symptoms = ['anxiety and nervousness', 'depression', 'shortness of breath', 'depressive or psychotic symptoms', 'sharp chest pain', 'dizziness', 'insomnia', 'abnormal involuntary movements', 'chest tightness', 'palpitations', 'irregular heartbeat', 'breathing fast', 'hoarse voice', 'sore throat', 'difficulty speaking', 'cough', 'nasal congestion', 'throat swelling', 'diminished hearing', 'lump in throat', 'throat feels tight', 'difficulty in swallowing', 'skin swelling', 'retention of urine', 'groin mass', 'leg pain', 'hip pain', 'suprapubic pain', 'blood in stool', 'lack of growth', 'emotional symptoms', 'elbow weakness', 'back weakness', 'pus in sputum', 'symptoms of the scrotum and testes', 'swelling of scrotum', 'pain in testicles', 'flatulence', 'pus draining from ear', 'jaundice', 'mass in scrotum', 'white discharge from eye', 'irritable infant', 'abusing alcohol', 'fainting', 'hostile behavior', 'drug abuse', 'sharp abdominal pain', 'feeling ill', 'vomiting', 'headache', 'nausea', 'diarrhea', 'vaginal itching', 'vaginal dryness', 'painful urination', 'involuntary urination', 'pain during intercourse', 'frequent urination', 'lower abdominal pain', 'vaginal discharge', 'blood in urine', 'hot flashes', 'intermenstrual bleeding', 'hand or finger pain', 'wrist pain', 'hand or finger swelling', 'arm pain', 'wrist swelling', 'arm stiffness or tightness', 'arm swelling', 'hand or finger stiffness or tightness', 'wrist stiffness or tightness', 'lip swelling', 'toothache', 'abnormal appearing skin', 'skin lesion', 'acne or pimples', 'dry lips', 'facial pain', 'mouth ulcer', 'skin growth', 'eye deviation', 'diminished vision', 'double vision', 'cross-eyed', 'symptoms of eye', 'pain in eye', 'eye moves abnormally', 'abnormal movement of eyelid', 'foreign body sensation in eye', 'irregular appearing scalp', 'swollen lymph nodes', 'back pain', 'neck pain', 'low back pain', 'pain of the anus', 'pain during pregnancy', 'pelvic pain', 'impotence', 'infant spitting up', 'vomiting blood', 'regurgitation', 'burning abdominal pain', 'restlessness', 'symptoms of infants', 'wheezing', 'peripheral edema', 'neck mass', 'ear pain', 'jaw swelling', 'mouth dryness', 'neck swelling', 'knee pain', 'foot or toe pain', 'bowlegged or knock-kneed', 'ankle pain', 'bones are painful', 'knee weakness', 'elbow pain', 'knee swelling', 'skin moles', 'knee lump or mass', 'weight gain', 'problems with movement', 'knee stiffness or tightness', 'leg swelling', 'foot or toe swelling', 'heartburn', 'smoking problems', 'muscle pain', 'infant feeding problem', 'recent weight loss', 'problems with shape or size of breast', 'underweight', 'difficulty eating', 'scanty menstrual flow', 'vaginal pain', 'vaginal redness', 'vulvar irritation', 'weakness', 'decreased heart rate', 'increased heart rate', 'bleeding or discharge from nipple', 'ringing in ear', 'plugged feeling in ear', 'itchy ear(s)', 'frontal headache', 'fluid in ear', 'neck stiffness or tightness', 'spots or clouds in vision', 'eye redness', 'lacrimation', 'itchiness of eye', 'blindness', 'eye burns or stings', 'itchy eyelid', 'feeling cold', 'decreased appetite', 'excessive appetite', 'excessive anger', 'loss of sensation', 'focal weakness', 'slurring words', 'symptoms of the face', 'disturbance of memory', 'paresthesia', 'side pain', 'fever', 'shoulder pain', 'shoulder stiffness or tightness', 'shoulder weakness', 'arm cramps or spasms', 'shoulder swelling', 'tongue lesions', 'leg cramps or spasms', 'abnormal appearing tongue', 'ache all over', 'lower body pain', 'problems during pregnancy', 'spotting or bleeding during pregnancy', 'cramps and spasms', 'upper abdominal pain', 'stomach bloating', 'changes in stool appearance', 'unusual color or odor to urine', 'kidney mass', 'swollen abdomen', 'symptoms of prostate', 'leg stiffness or tightness', 'difficulty breathing', 'rib pain', 'joint pain', 'muscle stiffness or tightness', 'pallor', 'hand or finger lump or mass', 'chills', 'groin pain', 'fatigue', 'abdominal distention', 'regurgitation.1', 'symptoms of the kidneys', 'melena', 'flushing', 'coughing up sputum', 'seizures', 'delusions or hallucinations', 'shoulder cramps or spasms', 'joint stiffness or tightness', 'pain or soreness of breast', 'excessive urination at night', 'bleeding from eye', 'rectal bleeding', 'constipation', 'temper problems', 'coryza', 'wrist weakness', 'eye strain', 'hemoptysis', 'lymphedema', 'skin on leg or foot looks infected', 'allergic reaction', 'congestion in chest', 'muscle swelling', 'pus in urine', 'abnormal size or shape of ear', 'low back weakness', 'sleepiness', 'apnea', 'abnormal breathing sounds', 'excessive growth', 'elbow cramps or spasms', 'feeling hot and cold', 'blood clots during menstrual periods', 'absence of menstruation', 'pulling at ears', 'gum pain', 'redness in ear', 'fluid retention', 'flu-like syndrome', 'sinus congestion', 'painful sinuses', 'fears and phobias', 'recent pregnancy', 'uterine contractions', 'burning chest pain', 'back cramps or spasms', 'stiffness all over', 'muscle cramps, contractures, or spasms', 'low back cramps or spasms', 'back mass or lump', 'nosebleed', 'long menstrual periods', 'heavy menstrual flow', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'frequent menstruation', 'sweating', 'mass on eyelid', 'swollen eye', 'eyelid swelling', 'eyelid lesion or rash', 'unwanted hair', 'symptoms of bladder', 'irregular appearing nails', 'itching of skin', 'hurts to breath', 'nailbiting', 'skin dryness, peeling, scaliness, or roughness', 'skin on arm or hand looks infected', 'skin irritation', 'itchy scalp', 'hip swelling', 'incontinence of stool', 'foot or toe cramps or spasms', 'warts', 'bumps on penis', 'too little hair', 'foot or toe lump or mass', 'skin rash', 'mass or swelling around the anus', 'low back swelling', 'ankle swelling', 'hip lump or mass', 'drainage in throat', 'dry or flaky scalp', 'premenstrual tension or irritability', 'feeling hot', 'feet turned in', 'foot or toe stiffness or tightness', 'pelvic pressure', 'elbow swelling', 'elbow stiffness or tightness', 'early or late onset of menopause', 'mass on ear', 'bleeding from ear', 'hand or finger weakness', 'low self-esteem', 'throat irritation', 'itching of the anus', 'swollen or red tonsils', 'irregular belly button', 'swollen tongue', 'lip sore', 'vulvar sore', 'hip stiffness or tightness', 'mouth pain', 'arm weakness', 'leg lump or mass', 'disturbance of smell or taste', 'discharge in stools', 'penis pain', 'loss of sex drive', 'obsessions and compulsions', 'antisocial behavior', 'neck cramps or spasms', 'pupils unequal', 'poor circulation', 'thirst', 'sleepwalking', 'skin oiliness', 'sneezing', 'bladder mass', 'knee cramps or spasms', 'premature ejaculation', 'leg weakness', 'posture problems', 'bleeding in mouth', 'tongue bleeding', 'change in skin mole size or color', 'penis redness', 'penile discharge', 'shoulder lump or mass', 'polyuria', 'cloudy eye', 'hysterical behavior', 'arm lump or mass', 'nightmares', 'bleeding gums', 'pain in gums', 'bedwetting', 'diaper rash', 'lump or mass of breast', 'vaginal bleeding after menopause', 'infrequent menstruation', 'mass on vulva', 'jaw pain', 'itching of scrotum', 'postpartum problems of the breast', 'eyelid retracted', 'hesitancy', 'elbow lump or mass', 'muscle weakness', 'throat redness', 'joint swelling', 'tongue pain', 'redness in or around nose', 'wrinkles on skin', 'foot or toe weakness', 'hand or finger cramps or spasms', 'back stiffness or tightness', 'wrist lump or mass', 'skin pain', 'low back stiffness or tightness', 'low urine output', 'skin on head or neck looks infected', 'stuttering or stammering', 'problems with orgasm', 'nose deformity', 'lump over jaw', 'sore in nose', 'hip weakness', 'back swelling', 'ankle stiffness or tightness', 'ankle weakness', 'neck weakness']


# -----------------------
# Helper functions for text cleaning and inference
def clean_text(text: str) -> str:
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return " ".join(text.translate(translator).split())

def classify_symptoms(text: str, threshold: float = 0.8) -> list[str]:
    inputs = initial_bert_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)
    logits = initial_bert_model(**inputs).logits
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    return [symptoms[i] for i, p in enumerate(probs[:len(symptoms)]) if p > threshold]

def predict_disease(detected: list[str], threshold: float = 0.995) -> list[tuple[str, float]]:
    txt = " ".join(detected)
    inputs = bert_tokenizer(
        txt, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)
    logits = bert_model(**inputs).logits
    probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
    candidates = [
        (disease_names[i], prob)
        for i, prob in enumerate(probs[:len(disease_names)])
        if prob > threshold
    ]
    return sorted(candidates, key=lambda x: x[1], reverse=True)

# -----------------------
# Gemma2LLM for generic QA and for routing decisions
class Gemma2LLM(BaseLLM):
    @property
    def _llm_type(self) -> str:
        return "gemma2"

    def _generate(
        self, prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None
    ) -> LLMResult:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        gens = []
        for prompt in prompts:
            payload = {"model": "gemma2-9b-it", "messages": [{"role": "user", "content": prompt}]}
            resp = requests.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            gens.append(ChatGeneration(message=AIMessage(content=content), generation_info={}))
        return LLMResult(generations=[[g] for g in gens])

llm_for_qa = Gemma2LLM()

# -----------------------
# FAISS and RetrievalQA setup for general queries
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_dir = "faiss_index"
if os.path.exists(index_dir) and os.listdir(index_dir):
    vector_store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
else:
    raise RuntimeError("FAISS index not found; build it first")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt_template = (
    "Use the following context to answer the question in markdown:\n(Only answer if the Question is Medical Related or else respond with 'I am a Medical Chatbot and is not specialized with other domain. Please stick with my specialized domain') \n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)
QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_for_qa,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# -----------------------
# LLM-based query routing function
def route_query_llm(text: str) -> dict:
    """
    Use the LLM to decide whether the query should be processed by the disease predictor or
    handled as a general query.
    
    The prompt instructs the LLM to return exactly one of two tokens:
    'disease_prediction' or 'general_query'
    """
    prompt = (
        "You are a routing agent. Given the following user query:\n\n"
        f"\"{text}\"\n\n"
        "Decide if this query is directly related to medical symptoms where user wants the disease to be predicted or if it is a general question. "
        "Respond only with one word: either 'disease_prediction' or 'general_query'."
    )
    try:
        # Generate the decision from the LLM.
        result = llm_for_qa._generate([prompt])
        decision = result.generations[0][0].message.content.strip().lower()
    except Exception as e:
        # Fallback to general query in case of any error.
        decision = "general_query"
    
    # If the answer is not exactly one of the options, default to general_query.
    if decision not in ["disease_prediction", "general_query"]:
        decision = "general_query"
    
    return {"branch": decision, "query": text}

# -----------------------
# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    # Use the LLM-based router to decide the branch.
    route_info = route_query_llm(user_input)
    # print(route_info["branch"])

    
    if route_info["branch"] == "disease_prediction":
        # First, clean text & classify symptoms.
        cleaned_text = clean_text(route_info["query"])
        detected_symptoms = classify_symptoms(cleaned_text)
        preds = predict_disease(detected_symptoms) if detected_symptoms else []
        
        if preds:
            # Build HTML list for diseases.
            diseases_list = "<ul>" + "".join(
                f"<li><strong>{d.upper()}</strong></li>" for d, _ in preds
            ) + "</ul>"
            descriptions = []
            for disease, _ in preds:
                info = retrieve_info_from_mongo(disease)
                if info["disease"]:
                    descriptions.append(markdown.markdown(info["description"].strip()) + "<br><br><br><br>")
            guidance_html = "".join(descriptions)
            collapsible_html = f"""
                <button class="toggle-btn" onclick="toggleDetails()">Details For Each Disease</button>
                <div id="disease-details" style="display: none; margin-top: 10px;">
                    {guidance_html}
                </div>
            """
            response_text = (
                f"Based on your symptoms, possible conditions are:<br>"
                f"{diseases_list}<br>{collapsible_html}"
            )
            return jsonify(response=response_text)
        else:
            # If no reliable symptom detection, fallback to general QA.
            qa_result = qa_chain.invoke({"query": cleaned_text})
            answer = qa_result.get("result", "I'm sorry, I couldn't find an answer to your question.")
            return jsonify(response=markdown.markdown(answer))
    
    elif route_info["branch"] == "general_query":
        qa_result = qa_chain.invoke({"query": route_info["query"]})
        answer = qa_result.get("result", "I'm sorry, I couldn't find an answer to your question.")
        return jsonify(response=markdown.markdown(answer))

if __name__ == "__main__":
    app.run(debug=True)