export const LOCAL_API_START_CMD = "bash haskell/scripts/run_api_with_db.sh";

export const localApiStartHelp = (options?: { multiline?: boolean }): string =>
  options?.multiline
    ? `Backend unreachable.\n\nStart it with:\n${LOCAL_API_START_CMD}`
    : `Backend unreachable. Start it with: ${LOCAL_API_START_CMD}`;
