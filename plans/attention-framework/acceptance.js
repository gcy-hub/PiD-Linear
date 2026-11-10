const ready = await runs.run("integration-dependency", {command:args.waitCommand, cwd:args.repo, rerun:true});
if (!ready.ok) throw new Error("Integration dependency failed: " + ready.output);
const result = await runs.run("acceptance", {agent:"worker", model:"zzz/gpt-6.1-sol:high", cwd:args.repo, task:args.brief + "\nIntegration receipt: " + ready.output + "\nReports: " + args.reports});
if (!result.ok) throw new Error("GPU acceptance/delivery failed: " + result.output);
return result.output;
