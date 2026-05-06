import fpdf
from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Phishing Detection Hybrid Model Report', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Intro
pdf.chapter_title('1. Datasets Used')
body1 = (
    "The hybrid deep learning model utilizes datasets composed of legitimate and phishing web pages for training and testing.\n\n"
    "Data structure:\n"
    "- 'features/legitimate_train.csv': Training features for legitimate URLs.\n"
    "- 'features/phish_train.csv': Training features for phishing URLs.\n"
    "- 'features/legitimate_test.csv': Testing features for legitimate URLs.\n"
    "- 'features/phish_test.csv': Testing features for phishing URLs.\n\n"
    "The models process two types of inputs:\n"
    "1. URL features: Extracted from the URL string itself by mapping characters to indices, producing sequences that are passed to an LSTM model (Model A).\n"
    "2. HTML features: Extracted from the web page's structure (e.g., number of hyperlinks, script tags, etc.) which are standardized and fed into a CNN model (Model B). "
    "These features undergo preprocessing like scaling using 'StandardScaler'."
)
pdf.chapter_body(body1)

# Metrics
pdf.chapter_title('2. Performance Metrics of the Hybrid Model (Model C)')
body2 = (
    "The test dataset contained a total of 1036 samples. "
    "Below are the performance metrics based on the exact confusion matrix:\n\n"
    "Confusion Matrix:\n"
    "[[486,  32]\n"
    " [ 34, 484]]\n\n"
    "Calculated Metrics:\n"
    "- Total Samples: 1036\n"
    "- True Negatives (Legitimate): 486\n"
    "- False Positives (Legitimate flagged as Phishing): 32\n"
    "- False Negatives (Phishing flagged as Legitimate): 34\n"
    "- True Positives (Phishing): 484\n\n"
    "- Accuracy: 93.63% (970 correct out of 1036)\n"
    "- Precision: 93.80% (Higher probability of a detected phish actually being phish)\n"
    "- Recall (Sensitivity): 93.44% (Percentage of actual phishing sites successfully detected)\n"
    "- F1-Score: 93.62% (Harmonic mean of precision and recall)\n\n"
    "These metrics demonstrate a high-performing hybrid and balanced model suitable for real-time phishing detection."
)
pdf.chapter_body(body2)

pdf.output('hybrid_model_performance.pdf')

print('PDF generated successfully.')
