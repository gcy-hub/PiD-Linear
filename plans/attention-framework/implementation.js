const call = (name, model, extra = {}) => ({agent: "worker", model, cwd: args.repo, task: args.briefs[name] + "\nReports: " + args.reports, ...extra});
const delivery = {type:"object", properties:{commit:{type:"string"}, checks:{type:"string"}}, required:["commit","checks"], additionalProperties:false};
let base = args.contract;
if (base !== undefined && !/^[0-9a-f]{40}$/.test(base)) throw new Error("args.contract must be a full committed SHA");
if (base === undefined) {
  const contract = await runs.run("C", call("C", "zzz/gpt-6.1-sol:high", {schema:delivery}));
  if (!contract.ok) throw new Error("C contract failed: " + contract.output);
  base = contract.data.commit;
}
const names = ["A", "M", "D", "T", "V"];
const results = await runs.all(names.map(name => ({key:name, ...call(name,
  name === "A" || name === "T" ? "zzz/gpt-6-astra:high" : "zzz/gpt-6.1-sol:high",
  {isolation:"worktree", schema:delivery, task:args.briefs[name] + "\nRequired base commit: " + base + ". Verify git merge-base HEAD " + base + " equals that SHA before editing. Read plans/attention-framework/contract.md. Main stays at this commit until all writers finish. Reports: " + args.reports})})));
if (results.some(r => !r.ok)) throw new Error("Parallel delivery failed: " + JSON.stringify(results));
const integrated = await runs.run("integrate", call("integrate", "zzz/gpt-6.1-sol:high", {schema:delivery,
  task:args.briefs.integrate + "\nContract: " + base + "\nDeliveries: " + JSON.stringify(results) + "\nReports: " + args.reports}));
if (!integrated.ok) throw new Error("Integration failed: " + integrated.output);
const cpu = await runs.run("cpu", call("cpu", "zzz/gpt-6.1-sol:high", {writer:false}));
if (!cpu.ok) throw new Error("CPU acceptance failed: " + cpu.output);
return {commit:integrated.data.commit, cpu:cpu.output};
