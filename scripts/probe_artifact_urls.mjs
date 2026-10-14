// 验证 `/api/jobs/{id}` 归一化后的**每个产物 URL** 是否真的可下载。
//
// 背景（用户报的 "Failed to load resource: 404"）：
//   原先服务端返回**绝对路径**（F:\3dgs-and-ai\workspace\...），前端把它拼进 URL，
//   绝对路径在工作区之外 ⇒ 必然 404；前端还有一处反斜杠正则 `/\\\\/g`
//   （匹配两个反斜杠）导致 `\` 被编成 %5C，双重叠加。
//   现在服务端返回 {path, rel, name, url}，本脚本就是它的回归检查：
//   要求每个 url 都能下到，且**不再出现 %5C / %3A**。
//
// 用法: node scripts/probe_artifact_urls.mjs [--port 8787] [--limit 12]
import fs from 'node:fs';

const arg = (k, d) => (process.argv.includes(k) ? process.argv[process.argv.indexOf(k) + 1] : d);
const port = arg('--port', '8787');
const limit = parseInt(arg('--limit', '12'), 10);
const base = `http://127.0.0.1:${port}`;

let jobs;
try {
  jobs = await (await fetch(`${base}/api/jobs`)).json();
} catch (e) {
  console.log(`**连不上服务** ${e.message}（先启动 scripts\\web.cmd）`);
  process.exit(2);
}
const list = [...(jobs.history || []), ...(jobs.running || [])];
console.log(`作业数: ${list.length}`);

let checked = 0, bad = 0, legacy = 0;
const problems = [];
for (const j of list.slice(0, limit)) {
  const jid = j.job_id || j;
  let d;
  try { d = await (await fetch(`${base}/api/jobs/${encodeURIComponent(jid)}`)).json(); }
  catch (e) { problems.push(`${jid}: 取详情失败 ${e.message}`); bad++; continue; }
  for (const [stage, v] of Object.entries(d.stages || {})) {
    for (const a of (v.artifacts || [])) {
      // 归一化后应是对象；字符串说明服务端没归一化（回归）
      if (typeof a === 'string' || !a.url) {
        legacy++;
        problems.push(`${jid}/${stage}: 产物未归一化 -> ${JSON.stringify(a).slice(0, 90)}`);
        bad++;
        continue;
      }
      checked++;
      if (/%5C|%3A/i.test(a.url)) {
        problems.push(`${jid}/${stage}: URL 里含 %5C/%3A -> ${a.url}`);
        bad++;
        continue;
      }
      let status = -1;
      try {
        const ctl = new AbortController();
        const resp = await fetch(base + a.url, { signal: ctl.signal });
        status = resp.status;
        const rd = resp.body && resp.body.getReader ? resp.body.getReader() : null;
        if (rd) { await rd.read(); ctl.abort(); }
      } catch { /* 中断读取属正常 */ }
      if (status !== 200) {
        problems.push(`${jid}/${stage}: HTTP ${status} -> ${a.url}`);
        bad++;
      }
    }
  }
}
console.log(`\n检查 ${checked} 个归一化产物 URL；未归一化的 ${legacy} 个；失败 ${bad} 个`);
for (const p of problems.slice(0, 10)) console.log(`  ${p}`);
console.log(bad ? '\n**仍有产物链接不可用 -> 浏览器会出现 404**'
                : '\n全部产物 URL 均为 200，且无 %5C/%3A 编码问题');
process.exitCode = bad ? 1 : 0;
