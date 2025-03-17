from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer

anonymizer = PresidioReversibleAnonymizer(
    analyzed_fields=["PERSON"],
)

anonymizer.anonymize("Me llamo Sofía") 

