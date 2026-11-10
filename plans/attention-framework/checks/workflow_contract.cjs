const fs = require('node:fs');
const assert = require('node:assert/strict');
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
const workflow = new AsyncFunction('args', 'runs', fs.readFileSync('plans/attention-framework/implementation.js', 'utf8'));
(async () => {
  for (const supplied of [true, false]) {
    const base = 'a'.repeat(40);
    const calls = [];
    const args = {repo: '.', reports: '/tmp', briefs: Object.fromEntries(['C','A','M','D','T','V','integrate','cpu'].map(n=>[n,n]))};
    if (supplied) args.contract = base;
    const runs = {
      run: async (name, options) => {
        calls.push(name);
        assert.equal(options.model, 'zzz/gpt-6.1-sol:high');
        return {ok:true, data:{commit:base}, output:'ok'};
      },
      all: async items => {
        assert.deepEqual(items.map(i=>i.key), ['A','M','D','T','V']);
        for (const i of items) {
          assert.equal(i.isolation, 'worktree');
          assert.equal(i.model, ['A','T'].includes(i.key) ? 'zzz/gpt-6-astra:high' : 'zzz/gpt-6.1-sol:high');
          assert.ok(i.task.includes(base));
        }
        return items.map(()=>({ok:true,data:{commit:base}}));
      }
    };
    assert.equal((await workflow(args,runs)).commit, base);
    assert.deepEqual(calls, supplied ? ['integrate','cpu'] : ['C','integrate','cpu']);
  }
  await assert.rejects(()=>workflow({contract:'short'}, {}), /full committed SHA/);
  console.log('Standalone contract skips C; legacy C path, five worktrees, exact models and integration preserved.');
})().catch(e=>{ console.error(e); process.exitCode=1; });
