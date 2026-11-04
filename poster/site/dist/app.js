const methods=[{id:'dpo',name:'DPO',color:'#203c4a',dash:''},{id:'grpo',name:'GRPO + decay',color:'#708087',dash:'7 4'},{id:'distill',name:'Self-distillation',color:'#547f2f',dash:'7 3 2 3'}];
const data=window.AUDIT_DATA;
const chart=document.querySelector('#curve-chart'), slider=document.querySelector('#hire-rate');
const fmt=n=>(n>0?'+':'')+n.toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
function draw(){
 const W=Math.max(300,chart.clientWidth),H=340,L=48,R=15,T=35,B=43,pw=W-L-R,ph=H-T-B;
 const xs=x=>L+(x-10)/25*pw,ys=y=>T+(5-y)/6.6*ph;
 const rate=Number(slider.value),index=Math.round((rate-10)*2),show=document.querySelector('#show-seeds').checked;
 let s=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><text x="${L}" y="15" fill="#536773" font-size="12">Δ unstable sets vs SFT (pp)</text>`;
 for(let y=-1;y<=4;y++)s+=`<line x1="${L}" x2="${W-R}" y1="${ys(y)}" y2="${ys(y)}" stroke="${y===0?'#85949c':'#e1e7e8'}"/><text x="${L-12}" y="${ys(y)+4}" text-anchor="end" font-size="12" fill="#536773">${y}</text>`;
 for(const x of [10,15,20,25,30,35])s+=`<text x="${xs(x)}" y="${H-23}" text-anchor="middle" font-size="12" fill="#536773">${x}</text>`;
 s+=`<text x="${L+pw/2}" y="${H-3}" text-anchor="middle" fill="#536773" font-size="12">Matched hire rate (%)</text><line x1="${xs(rate)}" x2="${xs(rate)}" y1="${T}" y2="${H-B}" stroke="#83929b" stroke-dasharray="3 4"/>`;
 const path=values=>values.map((v,i)=>(i?'L':'M')+xs(data.rates[i]).toFixed(2)+','+ys(v).toFixed(2)).join(' ');
 let readout='';
 for(const m of methods){const active=document.querySelector(`[data-method=${m.id}]`).checked;const d=data.methods[m.id];
  if(active){if(show)for(const seed of d.seeds)s+=`<path d="${path(seed)}" fill="none" stroke="${m.color}" stroke-width="1" opacity=".3" stroke-dasharray="${m.dash}"/>`;
  s+=`<path d="${path(d.mean)}" fill="none" stroke="${m.color}" stroke-width="2.8" stroke-dasharray="${m.dash}"/><circle cx="${xs(rate)}" cy="${ys(d.mean[index])}" r="4" fill="white" stroke="${m.color}" stroke-width="2"/>`;}
  readout+=`<div class="${active?'':'hidden-method'}"><span>${m.name}</span><strong>${fmt(d.mean[index])} pp</strong><span>${d.mean[index]>0?'More':'Fewer'} unstable sets</span></div>`;
 }
 chart.innerHTML=s+'</svg>';document.querySelector('#rate-value').value=rate.toFixed(1).replace('.0','')+'%';document.querySelector('#rate-readout').innerHTML=readout;
}
for(const input of document.querySelectorAll('input'))input.addEventListener('input',draw);
new ResizeObserver(draw).observe(chart);
document.querySelector('#seed-table tbody').innerHTML=methods.flatMap(m=>data.methods[m.id].intervals.map(row=>`<tr><td>${m.name}</td><td>${row.seed}</td><td>${fmt(row.delta)}</td><td>[${fmt(row.low)}, ${fmt(row.high)}]</td></tr>`)).join('');
document.querySelector('#overlap-table tbody').innerHTML=data.overlap.map(row=>`<tr><td>${row.method}</td><td>${row.seeds}</td><td>${row.actual.toFixed(3)}</td><td>${row.control.toFixed(3)}</td></tr>`).join('');
draw();
