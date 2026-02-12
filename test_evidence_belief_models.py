from evidence_belief_models import (
    ds_int_belief_model,
    ds_min_belief_model,
    sd_min_belief_model,
    get_key
)


 # Set of states
S = {'dp', 'sp', 'dm', 'sm', 'do'} 

# Quantitative evidence
E_quant = {
    get_key({'dp', 'dm', 'do'}): 0.9,
    get_key({'dm', 'sm'}): 0.75,
    get_key({'dp', 'sp'}): 0.45
}

# Qualitative evidence (same keys, uniform uncertainty)
E_qual = {
    get_key({'dp', 'dm', 'do'}): 0.5,
    get_key({'dm', 'sm'}): 0.5,
    get_key({'dp', 'sp'}): 0.5
}

# --- DS + intersection ---
result_int = ds_int_belief_model(S, E_quant)

print("\nDS-int belief model (DS justification frame + intersection allocation function):")
for k, b in result_int.items():
    print(f"belief for {k} -> {b:.2f}")
print("belief for other propositions -> 0")

# --- DS + minimal dense ---
result_min = ds_min_belief_model(S, E_quant)

print("\nDS-min belief model (DS justification frame + minimal dense set allocation function):")
for k, b in result_min.items():
    print(f"belief for {k} -> {b:.2f}")
print("belief for other propositions -> 0")

# --- SD + minimal dense (qualitative) ---
print("\nSD belief model (qualitative reasoning):")
result_sd = sd_min_belief_model(S, E_qual)

for k in result_sd:
    print(f"belief for {k}")

print("not belief for other propositions")