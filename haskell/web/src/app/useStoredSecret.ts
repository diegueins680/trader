import { useEffect, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { readLocalString, readSessionString, removeLocalKey, removeSessionKey, writeLocalString, writeSessionString } from "../lib/storage";

function readInitialSecret(key: string, persistSecrets: boolean): string {
  const persisted = readLocalString(key) ?? "";
  const session = readSessionString(key) ?? "";
  return persistSecrets ? persisted || session : session;
}

function persistSecretValue(key: string, persistSecrets: boolean, value: string) {
  const trimmed = value.trim();
  if (persistSecrets) {
    if (!trimmed) removeLocalKey(key);
    else writeLocalString(key, trimmed);
    removeSessionKey(key);
    return;
  }
  if (!trimmed) removeSessionKey(key);
  else writeSessionString(key, trimmed);
  removeLocalKey(key);
}

export function useStoredSecret(key: string, persistSecrets: boolean): [string, Dispatch<SetStateAction<string>>] {
  const [value, setValue] = useState<string>(() => readInitialSecret(key, persistSecrets));

  useEffect(() => {
    persistSecretValue(key, persistSecrets, value);
  }, [key, persistSecrets, value]);

  return [value, setValue];
}
