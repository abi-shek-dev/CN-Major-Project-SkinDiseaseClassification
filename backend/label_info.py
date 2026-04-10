CLASS_DETAILS = {
    "akiec": {
        "code": "akiec",
        "name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "description": "A sun-damage related lesion that can become cancerous and should be checked by a specialist.",
    },
    "bcc": {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "description": "A common form of skin cancer that usually grows slowly but still needs medical attention.",
    },
    "bkl": {
        "code": "bkl",
        "name": "Benign Keratosis-like Lesion",
        "description": "A non-cancerous lesion group that includes seborrheic keratoses and similar benign findings.",
    },
    "df": {
        "code": "df",
        "name": "Dermatofibroma",
        "description": "A usually harmless fibrous skin nodule that often appears firm and small.",
    },
    "mel": {
        "code": "mel",
        "name": "Melanoma",
        "description": "A serious type of skin cancer that requires urgent professional evaluation.",
    },
    "nv": {
        "code": "nv",
        "name": "Melanocytic Nevus",
        "description": "A common mole-like lesion that is often benign but can resemble other conditions.",
    },
    "vasc": {
        "code": "vasc",
        "name": "Vascular Lesion",
        "description": "A lesion related to blood vessels, such as angiomas or similar vascular findings.",
    },
}

DEFAULT_CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def describe_class(class_code):
    return CLASS_DETAILS.get(
        class_code,
        {
            "code": class_code,
            "name": class_code.upper(),
            "description": "No description available for this class.",
        },
    )
