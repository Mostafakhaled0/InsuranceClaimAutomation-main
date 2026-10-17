import os
import re
import json
from pathlib import Path
from difflib import SequenceMatcher

from flask import Flask, render_template, request
from PyPDF2 import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# Configuration
# =========================================================

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing. Set it in your environment first.")

app = Flask(__name__)

DOCUMENTS_DIR = Path("documents")
FAISS_PATH = Path("faiss")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0.4,
    max_tokens=2500,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

general_exclusion_list = [
    "HIV/AIDS",
    "Parkinson's disease",
    "Alzheimer's disease",
    "pregnancy",
    "substance abuse",
    "self-inflicted injuries",
    "sexually transmitted diseases",
    "pre-existing conditions",
]


# =========================================================
# Document Loading and Embeddings
# =========================================================

def get_document_loader():
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR.resolve()}"
        )

    loader = DirectoryLoader(
        str(DOCUMENTS_DIR),
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader,
        silent_errors=False,
    )

    documents = loader.load()
    if not documents:
        raise ValueError("No PDF documents were found in the documents folder.")
    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def get_embeddings():
    global vectorstore

    if FAISS_PATH.exists() and (FAISS_PATH / "index.faiss").exists():
        if vectorstore is None:
            vectorstore = FAISS.load_local(
                str(FAISS_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        return vectorstore

    documents = get_document_loader()
    chunks = get_text_chunks(documents)
    if not chunks:
        raise ValueError("No text could be extracted from the PDF documents.")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_PATH))
    return vectorstore


def get_retriever():
    return get_embeddings().as_retriever(search_kwargs={"k": 4})


def get_context(question):
    results = get_embeddings().similarity_search(question, k=4)
    return "\n\n".join(doc.page_content for doc in results)


def get_claim_approval_context():
    return get_context(
        "What documents and conditions are required for health insurance claim approval?"
    )


def get_general_exclusion_context():
    return get_context(
        "List the general exclusions and medical conditions not covered by this insurance policy."
    )


# =========================================================
# Medical Invoice Processing
# =========================================================

def get_file_content(file):
    if not file or not file.filename:
        raise ValueError("Please upload a medical invoice PDF.")
    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("The uploaded invoice must be a PDF file.")

    pdf = PdfReader(file)
    if pdf.is_encrypted:
        raise ValueError("The uploaded PDF is encrypted.")

    text_parts = []
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)

    text = "\n".join(text_parts).strip()
    if not text:
        raise ValueError(
            "No readable text was found in the PDF. "
            "The invoice may be a scanned image and require OCR."
        )
    return text


def get_bill_info(data):
    prompt = """
You extract information from medical invoices.

Extract:
1. The disease or medical condition stated in the invoice.
2. The total billed expense amount.

Return ONLY a valid JSON object in this exact format:
{"disease": "", "expense": ""}

Rules:
- The expense must be a numeric amount without currency symbols or commas.
- Do not invent missing information.
- If the disease cannot be identified, return an empty string.
- If the total expense cannot be identified, return an empty string.
- Do not include explanations outside the JSON object.
"""

    messages = [
        ("system", prompt),
        ("human", f"MEDICAL INVOICE:\n{data}"),
    ]

    response = llm.invoke(messages)
    content = response.content

    if not isinstance(content, str):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )

    content = content.strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

    try:
        bill_info = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini did not return valid JSON for the invoice.") from exc

    if not isinstance(bill_info, dict):
        raise ValueError("The invoice extraction result is invalid.")

    return {
        "disease": str(bill_info.get("disease") or "").strip(),
        "expense": str(bill_info.get("expense") or "").strip(),
    }


def parse_amount(value):
    if value is None:
        return None
    value = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if not match:
        return None
    try:
        amount = float(match.group())
    except ValueError:
        return None
    if amount < 0:
        return None
    return amount


# =========================================================
# Exclusion Checking (strengthened)
# =========================================================

def normalize_text(text):
    text = str(text or "").lower()
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


_STOPWORDS = {
    "disease", "diseases", "condition", "conditions",
    "treatment", "therapies", "therapy",
    "prescribed", "prescription", "medicines", "medicine", "medication",
    "and", "or", "for", "of", "with", "the", "a", "an", "to", "in",
}


def _meaningful_tokens(normalized: str) -> set:
    return {w for w in normalized.split() if w not in _STOPWORDS and len(w) > 1}


def find_excluded_disease(text):
    """
    Match text against general_exclusion_list using:
    1) exact normalized match
    2) substring match
    3) meaningful-token superset/subset match
    4) fuzzy similarity (difflib)
    """
    if not text:
        return None

    normalized = normalize_text(text)
    if not normalized:
        return None

    text_tokens = _meaningful_tokens(normalized)

    best_fuzzy_match = None
    best_fuzzy_score = 0.0

    for exclusion in general_exclusion_list:
        excl_norm = normalize_text(exclusion)

        # 1) Exact
        if normalized == excl_norm:
            return exclusion

        # 2) Substring (either direction)
        if excl_norm and (excl_norm in normalized or normalized in excl_norm):
            return exclusion

        # 3) Token-based
        excl_tokens = _meaningful_tokens(excl_norm)
        if text_tokens and excl_tokens:
            shared = text_tokens & excl_tokens
            # كل كلمات الاستثناء موجودة في النص (مثلاً "alzheimer" + "prescribed medicines")
            if excl_tokens.issubset(text_tokens):
                return exclusion
            # كل كلمات النص موجودة في الاستثناء
            if text_tokens.issubset(excl_tokens):
                return exclusion

        # 4) Fuzzy similarity على النص المطبع
        score = SequenceMatcher(None, normalized, excl_norm).ratio()
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_fuzzy_match = exclusion

    # Accept if very similar (handles typos like "Alzheimers" vs "Alzheimer's")
    if best_fuzzy_score >= 0.75:
        return best_fuzzy_match

    return None


# =========================================================
# Deterministic Rejection (bypasses the LLM entirely)
# =========================================================

def build_exclusion_rejection_report(
    patient_info: str,
    exclusion_match: str,
    max_amount: str,
) -> str:
    """
    Deterministic rejection report when a policy exclusion is matched.
    Does NOT call the LLM — guarantees EXCLUSION: FALSE and REJECTED.
    """
    return f"""INFORMATION: NOT ASSESSED (claim rejected on exclusion grounds)
EXCLUSION: FALSE
CLAIM DECISION: REJECTED

Executive Summary
The claim has been REJECTED because the reported medical condition matches
a policy exclusion. Per the insurance policy, excluded conditions are not
covered under any circumstances.

Introduction
This decision was produced deterministically by the claims rules engine.
When the reported diagnosis matches an entry in the policy's exclusion
list, the claim is automatically rejected. No further document review or
fraud analysis is required for the final outcome.

Claim Details
{patient_info}

Maximum Claim Amount Requested
{max_amount}

Exclusion Applied
{exclusion_match}

Reason for Decision
The policy explicitly excludes '{exclusion_match}' from coverage.
Because this condition is excluded, the claim cannot be approved,
regardless of the submitted documentation, invoice amount, or any
other information provided.

Note
This is a deterministic rejection based on the hardcoded policy exclusion
list. If you believe this exclusion does not apply to your specific case,
please contact the insurer directly to request a manual review by a human
claims adjuster.
"""


# =========================================================
# Claim Analysis Prompt (used only when NOT excluded)
# =========================================================

PROMPT = """
You are an AI assistant that prepares health insurance claim review reports.

Review the claim using the supplied policy context, patient information,
and medical invoice.

CLAIM APPROVAL REQUIREMENTS:
{claim_approval_context}

GENERAL EXCLUSIONS:
{general_exclusion_context}

PATIENT INFORMATION:
{patient_info}

MEDICAL BILL INFORMATION:
{medical_bill_info}

MAXIMUM CLAIM AMOUNT REQUESTED:
{max_amount}

Follow these rules:

1. INFORMATION:
Check the supplied information against the policy requirements.
If required information or documents are missing, mark INFORMATION as FALSE.
If the available evidence indicates all requirements are met, mark it TRUE.
If you cannot verify a requirement from the available evidence, state that
limitation clearly rather than claiming that the document was verified.

2. EXCLUSION:
Assess the medical condition against the supplied exclusion list.
If the condition is excluded, set EXCLUSION to FALSE and REJECT.
If evidence is insufficient, explicitly state that it needs human review.

3. DECISION:
Reject the claim if INFORMATION is FALSE or EXCLUSION is FALSE.
Do not approve a claim when required evidence is missing or a policy
exclusion applies.
The requested claim amount must not exceed the invoice amount.

4. FRAUD:
Report only specific inconsistencies supported by the submitted information.
Do not assert fraud without evidence.

Write the report using these headings:

INFORMATION: TRUE or FALSE
EXCLUSION: TRUE or FALSE
CLAIM DECISION: ACCEPTED, REJECTED, or NEEDS HUMAN REVIEW

Executive Summary
Introduction
Claim Details
Claim Description
Document Verification
Document Summary
Fraud and Inconsistency Checks
Reason for Decision

If accepted, state the maximum amount recommended for approval as
{max_amount}. Distinguish between information provided and information
actually verified from the supplied documents.
"""


prompt_template = PromptTemplate(
    input_variables=[
        "claim_approval_context",
        "general_exclusion_context",
        "patient_info",
        "medical_bill_info",
        "max_amount",
    ],
    template=PROMPT,
)


def generate_claim_report(patient_info, medical_bill_info, max_amount):
    claim_context = get_claim_approval_context()
    exclusion_context = get_general_exclusion_context()

    chain = prompt_template | llm

    response = chain.invoke({
        "claim_approval_context": claim_context,
        "general_exclusion_context": exclusion_context,
        "patient_info": patient_info,
        "medical_bill_info": medical_bill_info,
        "max_amount": max_amount,
    })

    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )

    return str(content)


# =========================================================
# Flask Routes
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    try:
        name = request.form.get("name", "").strip()
        address = request.form.get("address", "").strip()
        claim_type = request.form.get("claim_type", "").strip()
        claim_reason = request.form.get("claim_reason", "").strip()
        date = request.form.get("date", "").strip()
        medical_facility = request.form.get("medical_facility", "").strip()
        total_claim_amount = request.form.get("total_claim_amount", "").strip()
        description = request.form.get("description", "").strip()

        medical_bill = request.files.get("medical_bill")

        if not name or not claim_reason:
            raise ValueError("Please provide the patient's name and claim reason.")

        claim_amount = parse_amount(total_claim_amount)
        if claim_amount is None:
            raise ValueError("Please enter a valid claim amount.")
        if claim_amount == 0:
            raise ValueError("The claim amount must be greater than zero.")

        bill = get_file_content(medical_bill)
        bill_info = get_bill_info(bill)

        bill_expense = parse_amount(bill_info.get("expense"))
        disease = bill_info.get("disease", "")

        app.logger.info("Extracted disease=%r, expense=%r", disease, bill_expense)

        if bill_expense is None:
            output = (
                "The invoice amount could not be extracted reliably. "
                "Please upload a readable invoice and verify the amount."
            )
            return render_template(
                "result.html",
                name=name,
                address=address,
                claim_type=claim_type,
                claim_reason=claim_reason,
                date=date,
                medical_facility=medical_facility,
                total_claim_amount=total_claim_amount,
                description=description,
                output=output,
            )

        if claim_amount > bill_expense:
            output = (
                "CLAIM REJECTED: The requested claim amount exceeds "
                "the total expense extracted from the medical invoice. "
                f"Requested amount: {claim_amount:.2f}. "
                f"Invoice amount: {bill_expense:.2f}."
            )
            return render_template(
                "result.html",
                name=name,
                address=address,
                claim_type=claim_type,
                claim_reason=claim_reason,
                date=date,
                medical_facility=medical_facility,
                total_claim_amount=total_claim_amount,
                description=description,
                output=output,
            )

        patient_info = (
            f"Name: {name}\n"
            f"Address: {address}\n"
            f"Claim type: {claim_type}\n"
            f"Claim reason: {claim_reason}\n"
            f"Medical facility: {medical_facility}\n"
            f"Date: {date}\n"
            f"Total claim amount: {total_claim_amount}\n"
            f"Description: {description}"
        )

        # Try exclusion match on ALL relevant text sources.
        exclusion_match = (
            find_excluded_disease(disease)
            or find_excluded_disease(claim_reason)
            or find_excluded_disease(description)
        )

        app.logger.info("Exclusion match=%r", exclusion_match)

        # ---- DETERMINISTIC SHORT-CIRCUIT ----
        # If excluded, we don't ask the LLM at all.
        if exclusion_match:
            output = build_exclusion_rejection_report(
                patient_info=patient_info,
                exclusion_match=exclusion_match,
                max_amount=total_claim_amount,
            )
            return render_template(
                "result.html",
                name=name,
                address=address,
                claim_type=claim_type,
                claim_reason=claim_reason,
                date=date,
                medical_facility=medical_facility,
                total_claim_amount=total_claim_amount,
                description=description,
                output=output,
            )

        # ---- LLM PATH (only for non-excluded conditions) ----
        medical_bill_info = (
            f"Extracted disease: {disease or 'Not identified'}\n"
            f"Extracted invoice expense: {bill_expense:.2f}\n\n"
            f"Invoice text:\n{bill}"
        )

        output = generate_claim_report(
            patient_info=patient_info,
            medical_bill_info=medical_bill_info,
            max_amount=total_claim_amount,
        )

        return render_template(
            "result.html",
            name=name,
            address=address,
            claim_type=claim_type,
            claim_reason=claim_reason,
            date=date,
            medical_facility=medical_facility,
            total_claim_amount=total_claim_amount,
            description=description,
            output=output,
        )

    except Exception as exc:
        app.logger.exception("Claim processing failed")

        output = (
            "The claim could not be processed. "
            "Please check the server terminal for the technical error. "
            f"Details: {str(exc)}"
        )

        return render_template(
            "result.html",
            name=request.form.get("name", ""),
            address=request.form.get("address", ""),
            claim_type=request.form.get("claim_type", ""),
            claim_reason=request.form.get("claim_reason", ""),
            date=request.form.get("date", ""),
            medical_facility=request.form.get("medical_facility", ""),
            total_claim_amount=request.form.get("total_claim_amount", ""),
            description=request.form.get("description", ""),
            output=output,
        ), 500


# =========================================================
# Run Application
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=False)