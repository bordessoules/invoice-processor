import config

print("=== QWEN SAMPLING ===")
for k, v in config.SAMPLING_QWEN.items():
    print(f"  {k}: {v}")

print("\n=== GEMMA SAMPLING ===")
for k, v in config.SAMPLING_GEMMA.items():
    print(f"  {k}: {v}")

print("\n=== MODELES ===")
print(f"  QWEN  : {config.MODEL_QWEN}")
print(f"  GEMMA : {config.MODEL_GEMMA}")

print("\n=== PARAMETRES ENVOYES A CHAQUE APPEL ===")
print("  temperature         -> SAMPLING['temperature']")
print("  top_p               -> SAMPLING['top_p']")
print("  presence_penalty    -> SAMPLING['presence_penalty']  (defaut 0.0)")
print("  frequency_penalty   -> SAMPLING['frequency_penalty'] (defaut 0.0)")
print("  extra_body.top_k    -> SAMPLING['top_k'] si present")
print("  extra_body.min_p    -> SAMPLING['min_p'] si present")
print("  extra_body.repetition_penalty -> SAMPLING['repetition_penalty'] si present")
