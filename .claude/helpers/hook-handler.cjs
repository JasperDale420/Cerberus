// Minimal no-op hook handler for autoresearch and CI compatibility.
// The full hook handler is optional — this stub prevents MODULE_NOT_FOUND errors.
const action = process.argv[2] || 'unknown';
// Silent no-op for all actions
process.exit(0);
