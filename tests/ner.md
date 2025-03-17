# Methods that DON'T require NER:

- Pattern-based Masking


Uses regular expressions to identify structured data
Works for predictable formats like:

Phone numbers
Email addresses
Credit card numbers
IP addresses
Dates




- Statistical Anonymization


Applies techniques like k-anonymity or l-diversity
Works with numerical and categorical data
Doesn't need to understand named entities


- Token-based Redaction


Uses predefined dictionaries or word lists
Replaces specific terms without context understanding
Good for known sensitive terms


- Differential Privacy


Adds mathematical noise to data
Works at the statistical level
Doesn't require entity recognition

# Methods that DO use NER:

- Context-aware Anonymization


Needs NER to understand entities in context
Identifies relationships between entities
Maintains narrative coherence


- Semantic Anonymization


Uses NER for understanding entity types
Helps maintain text naturalness
Better handles ambiguous cases


- Intelligent Redaction


Uses NER for selective information hiding
Preserves relevant non-sensitive context
Better decision-making about what to anonymize


- підходи заміни (context-aware)
- уникнення викривлення результатів
- тести для семантики (зарплати, чисельні значення)
- приклад - вдвічі менше населення, прифронтова зона і тд
