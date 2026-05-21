import re

def applica_domain_enrichment(records):
    """
    Modulo condiviso: Domain Enrichment (Specifico per Fanuc)
    Riceve una lista di chunk puliti, costruisce la mappa degli allarmi 
    e inietta le procedure risolvendo le cross-reference.
    """
    print("🧠 Avvio Domain Enrichment (Risoluzione Cross-References)...")
    alarm_map = {}

    # 1. Popola mappa allarmi (Unendo il testo se l'allarme occupa più pagine!)
    for rec in records:
        if "SRVO-" in rec['title']:
            code = rec['title'].split(' ')[0]
            if code in alarm_map:
                alarm_map[code] += "\n" + rec['text']
            else:
                alarm_map[code] = rec['text']

    # 2. Iniezione Cross-References
    contatore_arricchimenti = 0
    for rec in records:
        # Cerca la dicitura esatta di rimando
        xref_match = re.search(r'same actions as (SRVO-\d+)', rec['text'])
        if xref_match:
            code_ref = xref_match.group(1)
            if code_ref in alarm_map:
                # Inietta il testo dell'allarme di riferimento in coda
                rec['text'] += f"\n\n[ENRICHMENT FROM {code_ref}]:\n" + alarm_map[code_ref]
                contatore_arricchimenti += 1
                print(f"    ⚓ Arricchito {rec['title']} con procedura da {code_ref}")

    print(f"✅ Domain Enrichment completato: {contatore_arricchimenti} cross-reference risolte.")
    
    return records