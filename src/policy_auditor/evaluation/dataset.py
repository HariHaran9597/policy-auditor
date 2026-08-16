"""Golden evaluation set — every answer is grounded in the sample Acme corpus.

Mix: table-lookup questions, cross-section questions, a cross-document
supersession question, and unanswerable questions that must be refused.
"""

EVAL_QUERIES: list[dict[str, str]] = [
    {
        "question": "How many working days of annual leave do employees get, and how many days can be carried over into the next year?",
        "ground_truth": "Employees get 24 working days of annual leave per year. Up to 8 unused days may be carried into the next leave year, and carried-over leave expires on 31 March.",
    },
    {
        "question": "How long is paternity leave and within how many months of the child's birth must it be used?",
        "ground_truth": "Paternity leave is 4 weeks and must be availed within 6 months of the child's birth, in blocks of no less than 1 week.",
    },
    {
        "question": "What is the per-night hotel ceiling for a Grade L3 employee travelling within India, and what flight class are they entitled to?",
        "ground_truth": "For Grade L3 the hotel ceiling is 12,000 INR per night within India, and the entitled flight class is Business.",
    },
    {
        "question": "What is the notice period when a Grade L3 employee resigns?",
        "ground_truth": "The notice period is 60 calendar days for Grades L3 and above (30 calendar days for L1 and L2).",
    },
    {
        "question": "Within how many days must the Internal Committee complete a POSH inquiry, and how quickly are complaints acknowledged?",
        "ground_truth": "Complaints are acknowledged within 2 working days. POSH complaints must have the inquiry completed within 90 days as required by law, and the IC's recommendations are implemented within 60 days of the report.",
    },
    {
        "question": "What is the minimum password length and the password rotation period?",
        "ground_truth": "The minimum password length is 12 characters, and passwords must be rotated every 180 days (immediately on suspicion of compromise).",
    },
    {
        "question": "After how many failed login attempts is an account locked, and for how long?",
        "ground_truth": "The account locks after 5 failed attempts, for 15 minutes.",
    },
    {
        "question": "What are the acknowledgement and resolution targets for a Sev2 security incident?",
        "ground_truth": "A Sev2 incident must be acknowledged within 30 minutes and resolved within 8 hours.",
    },
    {
        "question": "How quickly must a suspected personal-data breach be reported and to whom?",
        "ground_truth": "Suspected breaches must be reported to security@acme.example immediately upon suspicion and to the Data Protection Officer within 24 hours for statutory assessment; affected customers are notified per contractual terms.",
    },
    {
        "question": "How many days per week must an employee living within 50 km of an office work from that office?",
        "ground_truth": "At least two days per calendar week, unless an exception is approved in writing by the department Director.",
    },
    {
        "question": "As of March 2026, what is the monthly remote-work internet allowance for a Grade L1 fully-remote employee?",
        "ground_truth": "From 1 March 2026 the allowance is 1,200 INR per month for Grades L1 and L2 (Addendum A raised it from the 1,000 INR stated in the Handbook). Grade L3 remains at 1,000 INR.",
    },
    {
        "question": "Can Restricted data be accessed from a personal device?",
        "ground_truth": "No. Personal devices may access Internal-classification resources only through the managed browser; Confidential and Restricted resources require a corporate device.",
    },
    {
        "question": "How many hours per week of side employment cause the approval to be denied?",
        "ground_truth": "Approval for side employment is denied if the secondary occupation competes with Acme Corp or exceeds 20 hours per week.",
    },
    {
        "question": "What performance rating is required to be eligible for promotion, and what else must be true?",
        "ground_truth": "A rating of Meets Expectations or higher in the most recent cycle, completion of 12 months in the current grade, and a current-year promotion budget confirmation from Finance. Promotions are effective in the April and October cycles.",
    },
    {
        "question": "Can unused sick leave be carried into the next year?",
        "ground_truth": "No. Sick leave and casual leave cannot be carried over under any circumstances; only up to 8 days of annual leave can be carried over.",
    },
    {
        "question": "What is the tuition reimbursement for an employee's child's college education?",
        "ground_truth": "The policy documents do not specify any tuition reimbursement for employees' children; the only learning benefit is the 30,000 INR per year learning and development budget for the employee themselves (Grades L2 and above).",
    },
]
