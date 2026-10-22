"use strict";
const $ = (id) => document.getElementById(id);
const escapeHtml = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const number = (value, digits = 1) =>
  Number.isFinite(Number(value)) && value !== null && value !== ""
    ? Number(value).toLocaleString(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : "—";
const percent = (value) =>
  value === null || value === undefined
    ? "—"
    : `${number(Number(value) * 100, 1)}%`;
const date = (value) => (value ? String(value).slice(0, 10) : "—");
const state = { fighters: [], loaded: new Set(), recent: null, prediction: null, photos: new Map(), upcoming: null };
const directoryState={q:'',division:'',page:1};
let directoryRequest=0,profileRequest=0,directoryDebounce;
let eligibilityRequest=0,eligibilityDebounce,matchupSelectionVersion=0,comparisonInFlight=false;
state.matchupEligibility=null;
function directoryUrl() {
  const params=new URLSearchParams();
  if(directoryState.q)params.set('q',directoryState.q);
  if(directoryState.division)params.set('division',directoryState.division);
  if(directoryState.page>1)params.set('page',String(directoryState.page));
  return `/fighters${params.size?'?'+params:''}`;
}
function fighterProfileLink(id,name,className='') {
  return id?`<a class="${escapeHtml(className)}" data-fighter-link href="/fighters/${encodeURIComponent(id)}">${escapeHtml(name)}</a>`:escapeHtml(name);
}
async function loadDirectory() {
  const token=++directoryRequest,query={...directoryState};
  $('directory-query').value=query.q;
  $('directory-content').setAttribute('aria-busy','true');
  $('directory-content').innerHTML='<div class="panel loading-state">Finding covered athlete profiles…</div>';
  try {
    const params=new URLSearchParams({...query,per_page:'24'});
    const report=await api(`/api/v1/fighter-directory?${params}`);
    if(token!==directoryRequest)return;
    const rows=report.fighters||[],pagination=report.pagination||{};
    $('directory-title').textContent=`Browse ${number(report.total_canonical,0)} fighters`;
    $('directory-division').innerHTML='<option value="">All divisions</option>'+ (report.divisions||[]).map(d=>`<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join('');
    $('directory-division').value=query.division;
    const cards=rows.map(f=>{
      rememberPhoto(f);
      return `<a href="/fighters/${encodeURIComponent(f.fighter_id||f.id)}" data-fighter-link class="directory-card"><div class="directory-portrait">${athletePortrait(f,true)}<span class="directory-division">${escapeHtml(f.latest_division||'Division unavailable')}</span></div><div class="directory-card-copy"><h3>${escapeHtml(f.name)}</h3><div class="directory-card-stats"><span><b>${number(f.covered_bouts,0)}</b> covered ${f.covered_bouts===1?'bout':'bouts'}</span><span>Last covered<br><b>${escapeHtml(date(f.last_fight_date))}</b></span></div><span class="directory-card-action">View fighter profile <span aria-hidden="true">↗</span></span></div></a>`;
    }).join('');
    const total=pagination.total??rows.length;
$('directory-content').innerHTML=`<div class="directory-results-note"><span>${number(total,0)} ${total===1?'athlete':'athletes'}${query.q?` matching “${escapeHtml(query.q)}”`:''}</span><span>UFC history · UFC / ESPN athlete photos</span></div>${rows.length?`<div class="fighter-directory-grid">${cards}</div>`:`<div class="panel directory-empty"><p class="eyebrow">NO MATCHES</p><h3>No covered athlete matches these filters.</h3><p>Try a shorter name or choose all divisions.</p><button id="directory-clear" class="secondary-button" type="button">Clear filters</button></div>`}<nav class="directory-pagination" aria-label="Fighter directory pages"><button type="button" class="secondary-button" data-directory-page="${Math.max(1,query.page-1)}" ${pagination.has_previous?'':'disabled'}>← Previous</button><span>Page ${number(pagination.page??query.page,0)} of ${number(Math.max(1,pagination.total_pages||1),0)}</span><button type="button" class="secondary-button" data-directory-page="${query.page+1}" ${pagination.has_next?'':'disabled'}>Next →</button></nav>`;
    bindPortraits($('directory-content'));
    $('directory-content').querySelectorAll('[data-directory-page]').forEach(button=>button.addEventListener('click',()=>{
      directoryState.page=Number(button.dataset.directoryPage);history.pushState({},'',directoryUrl());loadDirectory();$('directory-form').scrollIntoView({block:'start',behavior:'instant'});
    }));
    $('directory-clear')?.addEventListener('click',()=>{directoryState.q='';directoryState.division='';directoryState.page=1;history.replaceState({},'',directoryUrl());loadDirectory();});
  } catch(error) {
    if(token!==directoryRequest)return;
    $('directory-content').innerHTML=`${errorPanel(error.message)}<button id="directory-retry" type="button" class="secondary-button">Retry directory</button>`;
    $('directory-retry').addEventListener('click',loadDirectory);
  } finally {if(token===directoryRequest)$('directory-content').setAttribute('aria-busy','false');}
}
function applyDirectoryFilters() {
  clearTimeout(directoryDebounce);
  directoryState.q=$('directory-query').value.trim();directoryState.division=$('directory-division').value;directoryState.page=1;
  history.replaceState({},'',directoryUrl());loadDirectory();
}
$('directory-form').addEventListener('submit',event=>{event.preventDefault();applyDirectoryFilters();});
$('directory-query').addEventListener('input',()=>{++directoryRequest;clearTimeout(directoryDebounce);directoryDebounce=setTimeout(applyDirectoryFilters,300);});
$('directory-division').addEventListener('change',applyDirectoryFilters);
function profileHistoryChart(items) {
  if(!items.length)return '<p class="panel-description">No observed count history is available.</p>';
  const rows=items.slice().reverse(),w=800,h=235,l=40,r=20,t=20,b=40;
  const max=Math.max(20,...rows.flatMap(row=>[Number(row.sig_landed)||0,Number(row.sig_absorbed)||0]));
  const scale=Math.ceil(max/50)*50,x=i=>l+(i+.5)/rows.length*(w-l-r),y=v=>h-b-v/scale*(h-t-b);
  const bar=Math.min(20,(w-l-r)/rows.length*.32);
  return `<svg class="profile-history-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Observed significant strikes landed and absorbed by bout, oldest to newest">${[0,.5,1].map(v=>`<line x1="${l}" x2="${w-r}" y1="${y(v*scale)}" y2="${y(v*scale)}" stroke="#37383f"/><text x="${l-7}" y="${y(v*scale)+4}" text-anchor="end" fill="#a7a9af" font-size="10">${number(v*scale,0)}</text>`).join('')}${rows.map((row,i)=>`<rect x="${x(i)-bar}" y="${y(row.sig_landed)}" width="${bar}" height="${h-b-y(row.sig_landed)}" fill="#f18a85"><title>${escapeHtml(row.event_date)} vs ${escapeHtml(row.opponent_name)}: ${number(row.sig_landed,0)} landed</title></rect><rect x="${x(i)}" y="${y(row.sig_absorbed)}" width="${bar}" height="${h-b-y(row.sig_absorbed)}" fill="#81b6f4"><title>${escapeHtml(row.event_date)} vs ${escapeHtml(row.opponent_name)}: ${number(row.sig_absorbed,0)} absorbed</title></rect>${rows.length<=8||i===0||i===rows.length-1||i%Math.ceil(rows.length/6)===0?`<text x="${x(i)}" y="${h-19}" text-anchor="middle" fill="#a7a9af" font-size="9">${escapeHtml(String(row.event_date).slice(2,7))}</text>`:''}`).join('')}<text x="${w/2}" y="${h-1}" text-anchor="middle" fill="#a7a9af" font-size="10">OBSERVED BOUTS · OLDEST → NEWEST</text></svg>`;
}
function renderProfileWindow(report, window) {
  const summary=report.windows?.[window]||{},all=report.history||[],rows=window==='last5'?all.slice(0,5):all;
  const definitions=[['sig_landed_per_min','LANDED / MIN','Significant-strike output',2],['sig_absorbed_per_min','ABSORBED / MIN','Lower means fewer absorbed',2],['net_sig_per_min','NET STRIKES / MIN','Landed minus absorbed',2],['sig_attempted_per_min','ATTEMPTS / MIN','Significant strikes attempted',2],['accuracy','ACCURACY','Landed ÷ attempted',1,true],['defense','DEFENSE','Opponent attempts avoided',1,true],['mean_duration_minutes','MEAN DURATION','Minutes per covered bout',1],['full_time_share','FULL-DURATION SHARE','Reached scheduled limit',1,true]];
  return `<div class="observed-heading"><div><p class="eyebrow">OBSERVED UFC HISTORY</p><h3>The striking profile</h3></div><div class="history-window" role="group" aria-label="Fighter profile sample"><button type="button" data-profile-window="last5" aria-pressed="${window==='last5'}">Last 5</button><button type="button" data-profile-window="career" aria-pressed="${window==='career'}">All covered</button></div></div><p class="panel-description">${number(summary.covered_bouts,0)} covered ${summary.covered_bouts===1?'bout':'bouts'} · ${number(summary.observed_minutes,1)} observed minutes.</p><div class="profile-stat-grid">${definitions.map(([key,label,note,digits,fraction])=>metric(label,fraction?percent(summary[key]):number(summary[key],digits),note)).join('')}</div><div class="panel data-panel profile-form-chart"><div class="profile-chart-title"><h3>Count history</h3><div class="comparison-legend"><span class="red-side">● Landed</span><span class="blue-side">● Absorbed</span></div></div>${profileHistoryChart(rows)}</div><div class="panel data-panel"><h3>${window==='last5'?'Last five covered bouts':'All covered bouts'}</h3><p class="panel-description">Newest first · Covered UFC bouts since 2013.</p>${rows.length?`<div class="table-scroll"><table><thead><tr><th>DATE / EVENT</th><th>OPPONENT</th><th>LANDED</th><th>ABSORBED</th><th>ATTEMPTED</th><th>MINUTES</th><th>FORMAT / METHOD</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${escapeHtml(date(row.event_date))}<small>${escapeHtml(row.event_name||'')}</small></td><td>${fighterProfileLink(row.opponent_id,row.opponent_name)}</td><td class="red-side">${number(row.sig_landed,0)}</td><td class="blue-side">${number(row.sig_absorbed,0)}</td><td>${number(row.sig_attempted,0)}</td><td>${number(row.duration_minutes,1)}</td><td>${escapeHtml(row.scheduled_rounds)} rounds<small>${escapeHtml(row.method||'Unavailable')}</small></td></tr>`).join('')}</tbody></table></div>`:'<p class="panel-description">No covered UFC bouts are available.</p>'}</div>`;
}
async function loadFighterProfile(id) {
  const token=++profileRequest;
  $('profile-content').innerHTML='<div class="panel loading-state">Loading this athlete’s covered fight history…</div>';
  $('profile-content').setAttribute('aria-busy','true');
  try {
    const report=await api(`/api/v1/fighters/${encodeURIComponent(id)}`);
    if(token!==profileRequest)return;
    const fighter=report.fighter||{},summary=report.windows?.career||{},photo=safeUrl(fighter.photo_source_url);
    rememberPhoto(fighter);document.title=`${fighter.name||'Fighter profile'} · OctogonML`;
$('profile-content').innerHTML=`<a class="text-link profile-back" data-fighter-link href="${escapeHtml(directoryUrl())}">← Fighter directory</a><div class="panel fighter-profile-hero"><div class="profile-portrait">${athletePortrait(fighter)}<span class="profile-photo-caption">${fighter.photo_url?'Official athlete photo':'Photo unavailable'}</span></div><div class="profile-identity"><p class="eyebrow">FIGHTER PROFILE · UFC HISTORY</p><h1>${escapeHtml(fighter.name)}</h1><div class="profile-tags"><span>Last covered division · ${escapeHtml(fighter.latest_division||'Unavailable')}</span><span>Last covered bout ${escapeHtml(date(fighter.last_fight_date))}</span></div><p class="profile-coverage">UFC bouts covered since 2013.</p><div class="profile-sample-stats"><div><strong>${number(fighter.covered_bouts??summary.covered_bouts,0)}</strong><span>Covered bouts</span></div><div><strong>${number(summary.observed_minutes,1)}</strong><span>Observed minutes</span></div></div><div class="profile-compare-actions"><button type="button" class="primary-button" data-profile-compare="fighter">Compare in red corner <span>↗</span></button><button type="button" class="secondary-button" data-profile-compare="opponent">Compare in blue corner <span>↗</span></button></div><p id="profile-compare-error" class="error-message" role="alert" hidden></p>${photo?`<a class="profile-source text-link" href="${escapeHtml(photo)}" target="_blank" rel="noopener">Photo source ↗</a>`:''}</div></div><section id="profile-observed" class="profile-observed" aria-label="Observed fighter statistics"></section><details class="panel data-panel profile-data-note"><summary>Coverage and source notes</summary><p class="panel-description">${escapeHtml(factText(report.note||'All values describe eligible completed bouts in this dataset. Missing history and photography remain explicitly unavailable.'))}</p><p class="panel-description">Stable athlete ID: ${escapeHtml(fighter.fighter_id||fighter.id)}. Photo source observed ${escapeHtml(date(fighter.photo_observed_at))}. Photography identifies the athlete and does not imply an affiliation with UFC or ESPN.</p></details>`;
    bindPortraits($('profile-content'));
    $('profile-content').querySelector('.profile-data-note').insertAdjacentHTML('beforeend',`<p class="panel-description">Dataset coverage: ${escapeHtml(date(report.data_window?.start))} → ${escapeHtml(date(report.data_window?.end))}. This athlete’s covered dates: ${escapeHtml(date(report.coverage?.fighter_start))} → ${escapeHtml(date(report.coverage?.fighter_end))}. Sources: ${escapeHtml((report.coverage?.source_names||[]).join(' · ')||'See dataset attribution')}.</p>`);
    const render=window=>{
      $('profile-observed').innerHTML=renderProfileWindow(report,window);
      $('profile-observed').querySelectorAll('[data-profile-window]').forEach(button=>button.addEventListener('click',()=>render(button.dataset.profileWindow)));
    };
    render('last5');
    $('profile-content').querySelectorAll('[data-profile-compare]').forEach(button=>button.addEventListener('click',async()=>{
      await initialization;
      const field=button.dataset.profileCompare,other=field==='fighter'?'opponent':'fighter';
      const known=state.fighters.find(row=>String(row.fighter_id)===String(id));
      if(!known){$('profile-compare-error').textContent='Matchup athlete selection is unavailable. Refresh when the athlete API is ready.';$('profile-compare-error').hidden=false;return;}
      const replacement=$(field).value!==String(id)?$(field).value:state.fighters.find(row=>String(row.fighter_id)!==String(id))?.fighter_id;
      const selectedOther=$(other).value===String(id)?replacement:$(other).value;
      for(const target of ['fighter','opponent']){
        $(`${target}-search`).value='';
        $(target).innerHTML=state.fighters.map(f=>`<option value="${escapeHtml(f.fighter_id)}">${escapeHtml(f.name)} · ${number(f.bout_count,0)} bouts</option>`).join('');
      }
      $(field).value=String(id);$(other).value=selectedOther;enforceDistinct();state.prediction=null;$('form-error').hidden=true;
      $('forecast-result').innerHTML=`<div class="panel empty-state"><p class="eyebrow">ATHLETE SELECTED</p><h2>${escapeHtml(known.name)} is ready to compare.</h2><p>Choose the other athlete, scheduled rounds, and hypothetical strike threshold, then select Compare fighters to generate both count forecasts.</p></div>`;
      navigateView('explorer');$('forecast-form').scrollIntoView({block:'start',behavior:'instant'});
    }));
  } catch(error){
    if(token!==profileRequest)return;
    $('profile-content').innerHTML=`<a class="text-link profile-back" data-fighter-link href="/fighters">← Fighter directory</a>${errorPanel(error.message)}<button id="profile-retry" class="secondary-button" type="button">Retry profile</button>`;
    $('profile-retry').addEventListener('click',()=>loadFighterProfile(id));
  } finally {if(token===profileRequest)$('profile-content').setAttribute('aria-busy','false');}
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      `The service returned an unreadable response (${response.status}).`,
    );
  }
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((x) => x.msg).join("; ")
          : detail?.message ? `${detail.message}${typeof detail.reason==='string'?`: ${detail.reason}`:''}` : data.message || `Request failed (${response.status}).`,
    );
  }
  return data;
}
function errorPanel(message) {
  return `<div class="panel data-panel"><h3>We couldn’t load this view.</h3><p class="error-message">${escapeHtml(message)}</p><p class="panel-description">Check the API service, then refresh to try again.</p></div>`;
}
function metric(label, value, note = "") {
  return `<div class="metric-card"><span class="stat-label">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></div>`;
}
function metricValues(data) {
  const m = data?.test || data?.holdout || data?.metrics || data || {};
  return {
    n: m.n ?? m.sample_size ?? m.count ?? m.n_predictions ?? m.athlete_count,
    mae: m.mae ?? m.mean_absolute_error,
    rmse: m.rmse,
    coverage: m.interval_coverage ?? m.coverage ?? m.coverage_80,
    level: m.interval_level ?? 0.8,
  };
}
function showView(name) {
    clearTimeout(directoryDebounce);
    if(name!=='profile')++profileRequest;
    document.querySelectorAll("[data-view]").forEach((b) => {
      const selected = b.dataset.view === (name==='profile'?'fighters':name);
      b.classList.toggle("active", selected);
      b.setAttribute("aria-selected", String(selected));
    });
    document
      .querySelectorAll(".view")
      .forEach(
        (view) => (view.hidden = view.id !== `view-${name}`),
      );
    document.querySelector(".note-number").textContent =
      { upcoming: "2026 / UFC", explorer: "MATCHUP / LAB", fighters:'FIGHTER / DIRECTORY', profile:'ATHLETE / PROFILE', recent: "RECENT / RESULTS", evidence: "MODEL / EVIDENCE", forward: "FORWARD / TRACKER" }[name]||'';
    document.querySelector('.intro').hidden=['fighters','profile'].includes(name);
    document.querySelector('.research-menu').open = false;
    if(name==='upcoming')loadUpcoming();
    if(name==='fighters')loadDirectory();
    if(name==='recent')loadRecent();
    if(name==='evidence')loadEvidence();
    if(name==='forward')loadForward();
    if(name!=='profile')document.title=name==='fighters'?'Fighters · OctogonML':'OctogonML · UFC strike forecasts';
}
function navigateView(name) {
  const path=name==='upcoming'?'/':name==='fighters'?directoryUrl():`/?view=${encodeURIComponent(name)}`;
  history.pushState({},'',path);showView(name);window.scrollTo({top:0,behavior:'instant'});
}
function routeLocation() {
  const match=location.pathname.match(/^\/fighters\/([^/]+)\/?$/);
  if(match){showView('profile');loadFighterProfile(decodeURIComponent(match[1]));return;}
  if(/^\/fighters\/?$/.test(location.pathname)){
    const query=new URLSearchParams(location.search);
    directoryState.q=query.get('q')||'';directoryState.division=query.get('division')||'';directoryState.page=Math.max(1,Number(query.get('page'))||1);
    showView('fighters');return;
  }
  const view=new URLSearchParams(location.search).get('view')||'upcoming';
  showView(['upcoming','explorer','recent','evidence','forward'].includes(view)?view:'upcoming');
}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>navigateView(button.dataset.view)));
window.addEventListener('popstate',routeLocation);
document.addEventListener('click',event=>{
  const link=event.target.closest('a[data-fighter-link]');
  if(!link||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey)return;
  event.preventDefault();history.pushState({},'',link.getAttribute('href'));routeLocation();window.scrollTo({top:0,behavior:'instant'});
});
async function initialize() {
  const results = await Promise.allSettled([
    api("/api/v1/overview"),
    api("/api/v1/fighters"),
    api("/api/v1/fighter-photos"),
    api("/api/v1/rankings"),
  ]);
  if(results[3].status==='fulfilled')state.rankings=results[3].value;
  if(results[2].status==='fulfilled'){
    const photos=results[2].value.fighters||{};
    ingestFighterPhotos(photos);
  }
  if (results[0].status === "fulfilled") {
    const overview = results[0].value;
    const d = overview.dataset || {},
      m = overview.model || {};
    $("summary-bouts").textContent = number(d.bout_count, 0);
    $("summary-fighters").textContent = number(d.fighter_count, 0);
    $("summary-date").textContent = date(d.date_end || d.data_cutoff);
    $("summary-model").textContent =
      m.model_version || m.current_version || m.version || "Available";
    if($("service-status"))$("service-status").textContent = "MODEL READY";
    if (d.source_names?.length)
      $("source-note").textContent =
        `Sources: ${d.source_names.join(" · ")} · Independent historical analysis.`;
  } else {
    if($("service-status"))$("service-status").textContent = "SERVICE UNAVAILABLE";
  }
  if (results[1].status === "fulfilled") {
    state.fighters = results[1].value.fighters || [];
    state.fighters.forEach(fighter => rememberPhoto(fighter));
    if (state.fighters.length < 2) {
      showFormError(
        "At least two athletes are required. Run the data pipeline before forecasting.",
      );
      return;
    }
    const options = state.fighters
      .map(
        (f) =>
          `<option value="${escapeHtml(f.fighter_id)}">${escapeHtml(f.name)}${f.bout_count !== undefined ? ` · ${f.bout_count} bouts` : ""}</option>`,
      )
      .join("");
    $("fighter").innerHTML = options;
    $("opponent").innerHTML = options;
    $("opponent").selectedIndex = 1;
    const preferredFighter = state.fighters.find(
        (f) => f.name === "Max Holloway",
      ),
      preferredOpponent = state.fighters.find((f) => f.name === "Ilia Topuria");
    if (preferredFighter && preferredOpponent) {
      $("fighter").value = preferredFighter.fighter_id;
      $("opponent").value = preferredOpponent.fighter_id;
    }
    ["fighter", "opponent", "forecast-button"].forEach(
      (id) => ($(id).disabled = false),
    );
    enforceDistinct();
  } else {
    showFormError(results[1].reason.message);
  }
}
function showFormError(message) {
  $("form-error").textContent = message;
  $("form-error").hidden = false;
}
function enforceDistinct(event) {
  if ($("fighter").value === $("opponent").value) {
    const target =
      event?.target?.id === "opponent" ? $("fighter") : $("opponent");
    target.value = state.fighters.find(
      (f) =>
        String(f.fighter_id) !== String((event?.target || $("fighter")).value),
    ).fighter_id;
  }
  for (const id of ["fighter", "opponent"])
    for (const option of $(id).options)
      option.disabled =
        option.value === $(id === "fighter" ? "opponent" : "fighter").value;
  refreshMatchupEligibility(Boolean(event));
}
function updateForecastButton() {
  $('forecast-button').disabled=comparisonInFlight||state.matchupEligibility?.status!=='compatible';
}
function clearComparisonForSelection() {
  state.prediction=null;state.matchupReport=null;$('form-error').hidden=true;
  if(state.reportDownloadUrl){URL.revokeObjectURL(state.reportDownloadUrl);state.reportDownloadUrl=null;}
  $('forecast-result').innerHTML='<div class="panel empty-state"><p class="eyebrow">NEW MATCHUP</p><h2>Choose your fighters.</h2><p>Once their divisions are checked, select Compare fighters for win probabilities and significant-strike forecasts.</p></div>';
}
function refreshMatchupEligibility(clearResults=true) {
  const token=++eligibilityRequest;++matchupSelectionVersion;clearTimeout(eligibilityDebounce);
  const fighter=$('fighter').value,opponent=$('opponent').value;
  state.matchupEligibility=null;updateForecastButton();
  if(clearResults)clearComparisonForSelection();
  ['fighter','opponent'].forEach(side=>$(side+'-division').textContent='Last covered division · Checking…');
  $('matchup-eligibility').className='matchup-eligibility';
  $('matchup-eligibility').innerHTML='<span class="eligibility-badge">CHECKING DIVISIONS</span><p>Based on last covered UFC division; same or neighboring divisions only.</p>';
  eligibilityDebounce=setTimeout(async()=>{
    try {
      const params=new URLSearchParams({fighter_id:fighter,opponent_id:opponent});
      const result=await api(`/api/v1/matchup-eligibility?${params}`);
      if(token!==eligibilityRequest)return;
      state.matchupEligibility=result;
      const compatible=result.status==='compatible';
      const label=compatible?(result.same_division?'Same division':'Neighboring divisions'):result.status==='unrealistic'?'Unrealistic matchup':'Division unavailable';
      $('fighter-division').textContent=`Last covered division · ${result.fighter?.division?.label||'Unavailable'}`;
      $('opponent-division').textContent=`Last covered division · ${result.opponent?.division?.label||'Unavailable'}`;
      $('matchup-eligibility').className=`matchup-eligibility ${compatible?'eligibility-compatible':'eligibility-blocked'}`;
      $('matchup-eligibility').innerHTML=`<span class="eligibility-badge">${escapeHtml(label)}</span><p>${escapeHtml(result.reason||'')}<small>Based on last covered UFC division; same or neighboring divisions only.</small></p>`;
    } catch(error) {
      if(token!==eligibilityRequest)return;
      $('matchup-eligibility').className='matchup-eligibility eligibility-blocked';
      $('matchup-eligibility').innerHTML=`<span class="eligibility-badge">Division check unavailable</span><p>Matchup compatibility could not be checked.</p><button class="secondary-button" type="button" id="retry-eligibility">Retry division check</button><details><summary>Service details</summary><p>${escapeHtml(error.message)}</p></details>`;
      $('retry-eligibility').addEventListener('click',()=>refreshMatchupEligibility(false));
      ['fighter','opponent'].forEach(side=>$(side+'-division').textContent='Last covered division · Unavailable');
    } finally {if(token===eligibilityRequest)updateForecastButton();}
  },150);
}
$("fighter").addEventListener("change", enforceDistinct);
$("opponent").addEventListener("change", enforceDistinct);
$("forecast-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("form-error").hidden = true;
  if(state.matchupEligibility?.status!=='compatible'){showFormError('Choose fighters in the same or neighboring covered UFC divisions before comparing.');updateForecastButton();return;}
  const selectionVersion=matchupSelectionVersion;
  const button = $("forecast-button");
  comparisonInFlight=true;
  button.disabled = true;
  button.textContent = "Calculating forecast…";
  $('forecast-result').setAttribute('aria-busy','true');
  try {
    const request = {
        fighter_id: $("fighter").value,
        opponent_id: $("opponent").value,
        scheduled_rounds: Number(
          document.querySelector("[name=scheduled_rounds]:checked").value,
        ),
        line: Number($("line").value),
      };
    const params=new URLSearchParams({fighter_id:request.fighter_id,opponent_id:request.opponent_id,scheduled_rounds:String(request.scheduled_rounds)});
    const results=await Promise.allSettled([
      api('/api/v1/predictions',{method:'POST',body:JSON.stringify(request)}),
      api('/api/v1/predictions',{method:'POST',body:JSON.stringify({...request,fighter_id:request.opponent_id,opponent_id:request.fighter_id})}),
      api(`/api/v1/matchups?${params}`),
      api('/api/v1/predict-winner',{method:'POST',body:JSON.stringify({fighter_a_id:request.fighter_id,fighter_b_id:request.opponent_id,scheduled_rounds:request.scheduled_rounds})}),
      api('/api/v1/predict-rounds',{method:'POST',body:JSON.stringify({fighter_a_id:request.fighter_id,fighter_b_id:request.opponent_id,scheduled_rounds:request.scheduled_rounds})}),
    ]);
    if(selectionVersion!==matchupSelectionVersion)return;
    if(results.every(result=>result.status==='rejected'))throw new Error(results[0].reason.message);
    const predictions=results.slice(0,2).map(result=>result.status==='fulfilled'?result.value:null);
    state.prediction=predictions[0]||predictions[1];
    renderComparison(predictions,results[2].status==='fulfilled'?results[2].value:null,request,results[3].status==='fulfilled'?results[3].value:null,results[3].status==='rejected'?results[3].reason.message:null,results.map((result,index)=>result.status==='rejected'?{service:['fighter_strikes','opponent_strikes','observed_context','winner','rounds'][index],message:result.reason.message}:null).filter(Boolean),results[4].status==='fulfilled'?results[4].value:null,results[4].status==='rejected'?results[4].reason.message:null);
    const failures=results.filter(result=>result.status==='rejected');
    if(failures.length)showFormError(`Some comparison data is unavailable: ${failures.map(result=>result.reason.message).join(' · ')}. Available results are shown below.`);
  } catch (error) {
    showFormError(error.message);
  } finally {
    comparisonInFlight=false;updateForecastButton();
    button.innerHTML = "Compare fighters <span>↗</span>";
    $('forecast-result').setAttribute('aria-busy','false');
  }
});
function comparisonDistribution(predictions) {
  const series=predictions.map(p=>(p?.distribution||[]).filter(d=>Number.isFinite(d.count)&&Number.isFinite(d.probability)));
  if(!series.some(s=>s.length))return '<p class="panel-description">Count-distribution detail is unavailable.</p>';
  const w=760,h=265,l=48,r=20,t=20,b=45;
  const maxX=Math.max(20,...series.flat().map(d=>d.count));
  const maxY=Math.max(.001,...series.flat().map(d=>d.probability));
  const x=v=>l+v/maxX*(w-l-r),y=v=>h-b-v/maxY*(h-t-b);
  const colors=['#f18a85','#81b6f4'];
  return `<svg class="comparison-pmf" viewBox="0 0 ${w} ${h}" role="img" aria-label="Two athlete significant-strike count probability distributions on identical axes">${[0,.25,.5,.75,1].map(v=>`<line x1="${l}" x2="${w-r}" y1="${y(v*maxY)}" y2="${y(v*maxY)}" stroke="#34424e"/><text x="${l-8}" y="${y(v*maxY)+4}" text-anchor="end" fill="#a2adb3" font-size="10">${number(v*maxY*100,1)}%</text><text x="${x(v*maxX)}" y="${h-24}" text-anchor="middle" fill="#a2adb3" font-size="10">${number(v*maxX,0)}</text>`).join('')}${series.map((s,i)=>s.length?`<path d="M${x(0)},${y(0)} ${s.map(d=>`L${x(d.count)},${y(d.probability)}`).join(' ')} L${x(s.at(-1).count)},${y(0)} Z" fill="${colors[i]}" fill-opacity=".12"/><polyline points="${s.map(d=>`${x(d.count)},${y(d.probability)}`).join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="2"/>${predictions[i]?`<line x1="${x(predictions[i].mean)}" x2="${x(predictions[i].mean)}" y1="${t}" y2="${h-b}" stroke="${colors[i]}" stroke-dasharray="4 5"/>`:''}`:'').join('')}<text x="${w/2}" y="${h-3}" text-anchor="middle" fill="#a2adb3" font-size="10">SIGNIFICANT STRIKES LANDED · FULL BOUT</text></svg>`;
}
function currentRankBadge(id) {
  const snapshot=state.rankings;if(!snapshot)return '';
  const rows=(snapshot.entries||[]).filter(row=>String(row.fighter_id)===String(id));
  if(!rows.length)return '';
  return `<div class="observed-rank-badges">${rows.map(row=>`<span>${escapeHtml(row.division)} ${row.is_champion?'Champion':`#${escapeHtml(row.rank)}`}</span>`).join('')}<small>Observed ${escapeHtml(date(snapshot.observed_at))} · <a href="${safeUrl(snapshot.source_url)||'/api/v1/rankings'}" target="_blank" rel="noopener">Official source ↗</a></small></div>`;
}
function fundamentalsPanel(report,athletes) {
  const context=report?.fighter_context;if(!context)return '';
  const sides=[context.fighter||{},context.opponent||{}];
  const fmt=(value,digits=1)=>Number.isFinite(value)?number(value,digits):'—';
  const metrics=[['Age','age',1,1,'years'],['Height','height_cm',1,1,'cm'],['Reach','reach_cm',1,1,'cm'],['Takedowns landed','last5_td_landed_per_minute',15,1,'/ 15 min'],['Submission attempts','last5_sub_attempted_per_minute',15,1,'/ 15 min'],['Control time','last5_control_seconds_per_minute',.25,1,'min / 15 min'],['Knockdowns','last5_knockdowns_per_minute',15,2,'/ 15 min']];
  return `<section class="panel fundamentals-panel"><div class="winner-heading"><div><p class="eyebrow">MATCHUP CONTEXT</p><h3>Matchup fundamentals</h3></div><span class="winner-scenario">Last 5 covered bouts · Grappling</span></div><div class="fundamentals-names"><strong class="red-side">${escapeHtml(athletes[0].name)}</strong><strong class="blue-side">${escapeHtml(athletes[1].name)}</strong></div><div class="fundamentals-grid">${metrics.map(([label,key,multiplier,digits,unit])=>`<div><strong class="red-side">${fmt(Number.isFinite(sides[0][key])?sides[0][key]*multiplier:null,digits)}</strong><span>${label}<small>${unit}</small></span><strong class="blue-side">${fmt(Number.isFinite(sides[1][key])?sides[1][key]*multiplier:null,digits)}</strong></div>`).join('')}</div><p class="panel-description">Grappling rates use each athlete’s last ${sides.map(side=>Math.min(5,side.grappling_prior_bouts||0)).join(' / ')} covered bouts, normalized by observed fight time. — means unavailable.</p><details class="winner-details"><summary>Ranking inputs & coverage</summary><p class="panel-description">The latest observed rankings displayed by athlete names are a separate official snapshot, not a historical model input. The model uses dated ranking observations only when sufficiently recent; missing or stale rankings remain unknown. The official display snapshot has no explicit weekly publication date.</p>${sides.map((side,i)=>`<p class="panel-description"><strong>${escapeHtml(athletes[i].name)}</strong> · Model ranking ${side.rank_missing||!Number.isFinite(side.rank_observed)?'Unknown':side.champion_observed?'Champion':`#${number(side.rank_observed,0)}`} · Board date ${escapeHtml(side.ranking_board_date||'Unavailable')} · Board age ${fmt(side.rank_board_age_days,0)} days · Grappling sample ${number(side.grappling_prior_bouts,0)} covered bouts.</p>`).join('')}<a class="text-link" href="/api/v1/rankings">Inspect latest observed ranking snapshot ↗</a></details></section>`;
}
function winnerPanel(report,athletes,error) {
  const a=report?.p_fighter_win,b=report?.p_opponent_win;
  const valid=Number.isFinite(a)&&Number.isFinite(b)&&a>=0&&a<=1&&b>=0&&b<=1&&Math.abs(a+b-1)<1e-6&&String(report.fighter?.id)===String(athletes[0].fighter_id)&&String(report.opponent?.id)===String(athletes[1].fighter_id);
  if(!valid)return `<section class="panel winner-panel winner-unavailable" aria-label="Win probability"><div class="winner-heading"><div><p class="eyebrow">OUTCOME MODEL · ML BASELINE</p><h3>Win probability</h3></div><span class="pill">UNAVAILABLE</span></div><p class="panel-description">Win probabilities are unavailable for this matchup. Strike forecasts and observed history are shown independently.</p>${error?`<details class="winner-details"><summary>Service details</summary><p class="panel-description">${escapeHtml(error)}</p></details>`:''}</section>`;
  const evaluation=report.evaluation||{};
  return `<section class="panel winner-panel" aria-label="Win probability"><div class="winner-heading"><div><p class="eyebrow">OUTCOME MODEL · ML BASELINE</p><h3>Win probability</h3></div><span class="winner-scenario">${escapeHtml(report.scheduled_rounds)} scheduled rounds</span></div><div class="winner-sides"><div class="red-side"><span>${fighterProfileLink(athletes[0].fighter_id,athletes[0].name)}</span><strong>${percent(a)}</strong></div><div class="blue-side"><span>${fighterProfileLink(athletes[1].fighter_id,athletes[1].name)}</span><strong>${percent(b)}</strong></div></div><div class="winner-probability-bar" role="img" aria-label="${escapeHtml(athletes[0].name)} ${percent(a)}, ${escapeHtml(athletes[1].name)} ${percent(b)} win probability"><span class="winner-red" style="width:${a*100}%"></span><span class="winner-blue" style="width:${b*100}%"></span></div><details class="winner-details"><summary>Among decisive outcomes <span>· Model details</span></summary><p class="panel-description">Probabilities are conditional on a decisive outcome; draws and no-contests are excluded. This separate outcome classifier does not derive a winner from projected strike totals. These are model estimates, not market odds.</p><div class="winner-evaluation"><span>Evaluation ${escapeHtml(Array.isArray(evaluation.period)?evaluation.period.join(' → '):evaluation.period||'See model report')} · ${escapeHtml(factText(evaluation.kind||'Historical evaluation'))}</span><span>${number(evaluation.n_bouts,0)} bouts · Log loss ${number(evaluation.log_loss,4)} · Brier ${number(evaluation.brier_score,4)} · Accuracy ${percent(evaluation.accuracy)}</span></div>${report.warnings?.length?`<ul class="limitation-list">${report.warnings.map(w=>`<li>${escapeHtml(w)}</li>`).join('')}</ul>`:''}<dl class="metadata-list">${metadataRow('Prediction ID',report.prediction_id)}${metadataRow('Model version',report.model_version)}${metadataRow('Feature cutoff',date(report.feature_cutoff))}${metadataRow('Data cutoff',date(report.data_cutoff))}${metadataRow('Forecast date',date(report.forecast_date))}${metadataRow('Covered history',`${report.history?.fighter_bouts??'—'} / ${report.history?.opponent_bouts??'—'} bouts`)}${metadataRow('Artifact SHA256',report.artifact_sha256)}${metadataRow('Source snapshot SHA256',report.source_snapshot_sha256)}${metadataRow('Outcome labels SHA256',report.outcome_labels_sha256)}</dl><a class="text-link" href="/api/v1/winner-model">Inspect outcome model report ↗</a></details></section>`;
}
function roundsPanel(report,athletes,request,error) {
  const probability=value=>Number.isFinite(value)&&value>=0&&value<=1;
  const valid=report&&String(report.fighter?.id)===String(athletes[0].fighter_id)&&String(report.opponent?.id)===String(athletes[1].fighter_id)&&report.scheduled_rounds===request.scheduled_rounds&&Number.isFinite(report.expected_duration_minutes)&&report.expected_duration_minutes>=0&&report.expected_duration_minutes<=request.scheduled_rounds*5&&probability(report.p_complete_scheduled_time)&&probability(report.p_finish_early)&&Math.abs(report.p_complete_scheduled_time+report.p_finish_early-1)<1e-6;
  if(!valid)return `<section class="panel rounds-panel"><div class="winner-heading"><div><p class="eyebrow">FIGHT TIME MODEL</p><h3>Rounds & duration</h3></div><span class="pill">UNAVAILABLE</span></div><p class="panel-description">Elapsed-time probabilities are unavailable for this matchup. Other models are shown independently.</p>${error?`<details><summary>Service details</summary><p class="panel-description">${escapeHtml(error)}</p></details>`:''}</section>`;
  const thresholds=(report.round_thresholds||[]).filter(row=>Number.isFinite(row.line)&&row.line>=1&&row.line<request.scheduled_rounds&&Number.isFinite(row.minutes)&&[row.p_over,row.p_under,row.p_push].every(probability)&&Math.abs(row.p_over+row.p_under+row.p_push-1)<1e-6);
  const evaluation=report.evaluation||{};
  const reach=(report.round_reach||[]).filter(row=>Number.isInteger(row.round)&&row.round>=1&&row.round<=request.scheduled_rounds&&probability(row.probability));
  return `<section class="panel rounds-panel" aria-label="Rounds and duration probabilities"><div class="winner-heading"><div><p class="eyebrow">FIGHT TIME MODEL</p><h3>Rounds & duration</h3></div><span class="winner-scenario">${escapeHtml(request.scheduled_rounds)} scheduled rounds</span></div><div class="duration-highlights"><div><span>Completes scheduled time</span><strong>${percent(report.p_complete_scheduled_time)}</strong><small>Full ${request.scheduled_rounds*5} minutes</small></div><div><span>Finishes early</span><strong>${percent(report.p_finish_early)}</strong><small>Before scheduled time</small></div><div><span>Estimated duration</span><strong>${number(report.expected_duration_minutes,1)}<small> min</small></strong><small>Approximate model mean</small></div></div><div class="round-reach">${reach.map(row=>`<div><span>Starts round ${row.round}</span><div class="reach-track"><i style="width:${row.probability*100}%"></i></div><strong>${percent(row.probability)}</strong></div>`).join('')}</div><div class="table-scroll"><table class="round-threshold-table"><caption>Elapsed round thresholds</caption><thead><tr><th>Round line</th><th>Time</th><th>Over</th><th>Under</th><th>Exact boundary</th></tr></thead><tbody>${thresholds.map(row=>`<tr><th>${number(row.line,1)}</th><td>${Math.floor(row.minutes)}:${String(Math.round((row.minutes%1)*60)).padStart(2,'0')}</td><td>${percent(row.p_over)}</td><td>${percent(row.p_under)}</td><td>${percent(row.p_push)}</td></tr>`).join('')}</tbody></table></div><p class="panel-description">Elapsed fight time. Over 1.5 means lasting longer than 7:30; exact boundaries are listed separately. These are not market settlement rules. Completing scheduled time does not necessarily mean a decision.</p><details class="winner-details"><summary>Time model details & provenance</summary><p class="panel-description">A separate duration model estimates these probabilities independently of strike totals and the winner model. ${escapeHtml(report.mean_approximation||'Expected minutes are an approximation.')}</p><p class="panel-description">Evaluation ${escapeHtml(Array.isArray(evaluation.period)?evaluation.period.join(' → '):evaluation.period||'See time model report')} · ${escapeHtml(factText(evaluation.kind||'Historical evaluation'))}. ${number(evaluation.n_bouts,0)} bouts · Category log loss ${number(evaluation.category_log_loss,4)} · Survival Brier ${number(evaluation.survival_brier,4)} · Completion Brier ${number(evaluation.completion_brier,4)}.</p>${report.warnings?.length?`<ul class="limitation-list">${report.warnings.map(w=>`<li>${escapeHtml(w)}</li>`).join('')}</ul>`:''}<dl class="metadata-list">${metadataRow('Prediction ID',report.prediction_id)}${metadataRow('Model version',report.model_version)}${metadataRow('Feature cutoff',date(report.feature_cutoff))}${metadataRow('Data cutoff',date(report.data_cutoff))}${metadataRow('Artifact SHA256',report.artifact_sha256)}${metadataRow('Source snapshot SHA256',report.source_snapshot_sha256)}${metadataRow('Duration labels SHA256',report.duration_labels_sha256)}</dl><a class="text-link" href="/api/v1/rounds-model">Inspect time model report ↗</a></details></section>`;
}
function renderComparison(predictions, context, request, winner=null, winnerError=null, failures=[], rounds=null, roundsError=null) {
  const athletes=[request.fighter_id,request.opponent_id].map(id=>state.fighters.find(f=>String(f.fighter_id)===String(id))||{fighter_id:id,name:'Athlete'});
const card=(athlete,p,i)=>`<article class="comparison-athlete ${i?'blue-side':'red-side'}">${athletePortrait({...athlete,id:athlete.fighter_id},true)}<p class="eyebrow">${i?'BLUE':'RED'} SIDE</p><h2>${fighterProfileLink(athlete.fighter_id,athlete.name)}</h2>${currentRankBadge(athlete.fighter_id)}${p?`<div class="comparison-estimate"><strong>${number(p.mean)}</strong><span>Expected significant strikes</span></div><div class="comparison-range"><span>Median <b>${number(p.median,0)}</b></span><span>${percent(p.interval_level??.8)} range <b>${number(p.interval_low,0)}–${number(p.interval_high,0)}</b></span></div><details class="comparison-probabilities"><summary>Probabilities at ${number(p.line)} strikes</summary><p>Over ${percent(p.p_over)} · Under ${percent(p.p_under)} · Push ${percent(p.p_push)}</p><small>Same hypothetical threshold for each athlete; not a market line.</small></details>`:'<p class="error-message">Forecast unavailable. Submit again to retry.</p>'}</article>`;
  $('forecast-result').innerHTML=`<div class="panel comparison-projections"><div class="comparison-title"><div><p class="eyebrow">MODEL PROJECTIONS</p><h3>${escapeHtml(request.scheduled_rounds)}-round matchup</h3></div><span class="pill">HYPOTHETICAL SCENARIO</span></div><div class="comparison-hero">${card(athletes[0],predictions[0],0)}<span class="comparison-versus">VS</span>${card(athletes[1],predictions[1],1)}</div><div class="comparison-chart"><h3>How the count outcomes compare</h3><p class="panel-description">Solid curves show count probabilities; dashed lines mark expected counts. Overlap reflects uncertainty, not a win probability.</p><div class="comparison-legend"><span class="red-side">● ${escapeHtml(athletes[0].name)}</span><span class="blue-side">● ${escapeHtml(athletes[1].name)}</span></div>${comparisonDistribution(predictions)}<p class="panel-description">Displayed PMFs are finite excerpts. Remaining probability beyond the chart: ${predictions.map((p,i)=>`${escapeHtml(athletes[i].name)} ${p?percent(p.tail_probability??p.distribution_tail_probability):'—'}`).join(' · ')}.</p></div><details class="forecast-notes"><summary>Forecast warnings and provenance</summary>${predictions.map((p,i)=>p?`<h4>${escapeHtml(athletes[i].name)}</h4><ul class="warnings">${(p.warnings||[]).map(w=>`<li>${escapeHtml(w)}</li>`).join('')}</ul><p class="provenance">Model ${escapeHtml(p.model_version)} · ID ${escapeHtml(p.prediction_id)}<br>Features through ${escapeHtml(date(p.feature_cutoff))} · Data through ${escapeHtml(date(p.data_cutoff))}</p>`:'').join('')}</details></div><div id="comparison-observed" class="panel data-panel"></div>`;
  bindPortraits($('forecast-result'));
  $('forecast-result').insertAdjacentHTML('afterbegin',fundamentalsPanel(winner,athletes));
  $('forecast-result').insertAdjacentHTML('afterbegin',roundsPanel(rounds,athletes,request,roundsError));
  $('forecast-result').insertAdjacentHTML('afterbegin',winnerPanel(winner,athletes,winnerError));
  $('forecast-result').insertAdjacentHTML('afterbegin','<aside class="forecast-use-notice" role="note"><strong>Research only · Not betting advice</strong><p>Model estimates can be wrong. Do not use OctogonML to place bets or risk money. You are responsible for how you use these estimates.</p></aside>');
  const hero=$('forecast-result').querySelector('.comparison-hero');
  hero.classList.add('comparison-stage');
  hero.insertBefore($('comparison-observed'),hero.lastElementChild);
  const recentHistories=['fighter','opponent'].map(side=>(context?.comparison_dashboard?.history?.[side]||[]).slice(0,5));
  const recentMax=Math.max(1,...recentHistories.flat().map(row=>Number(row.sig_landed)||0));
  hero.querySelectorAll('.comparison-athlete').forEach((card,index)=>{
    const rows=recentHistories[index];
    if(!rows.length)return;
    card.insertAdjacentHTML('beforeend',`<div class="comparison-recent"><h4>Last ${rows.length} observed counts</h4><svg viewBox="0 0 220 100" role="img" aria-label="Recent observed significant strikes, newest first, shared scale across athletes">${rows.map((row,i)=>`<rect x="${i*44+8}" y="${68-Number(row.sig_landed)/recentMax*52}" width="28" height="${Number(row.sig_landed)/recentMax*52}" fill="currentColor" fill-opacity=".7"><title>${escapeHtml(row.event_date)} vs ${escapeHtml(row.opponent_name)}: ${number(row.sig_landed,0)} landed</title></rect><text x="${i*44+22}" y="${Math.max(10,63-Number(row.sig_landed)/recentMax*52)}" fill="currentColor" text-anchor="middle" font-size="10">${number(row.sig_landed,0)}</text><text x="${i*44+22}" y="85" fill="#a7a9af" text-anchor="middle" font-size="8">${escapeHtml(String(row.event_date).slice(2,7))}</text>`).join('')}</svg><p>Actual landed · newest first</p></div>`);
  });
  const dashboard=context?.comparison_dashboard;
  if(dashboard){
    const renderWindow=window=>{
      $('comparison-observed').innerHTML=renderObservedComparison(dashboard,window,athletes);
      $('comparison-observed').querySelectorAll('[data-history-window]').forEach(button=>button.addEventListener('click',()=>renderWindow(button.dataset.historyWindow)));
    };
    renderWindow('last5');
  }else {$('comparison-observed').innerHTML='<h3>Observed history unavailable</h3><p class="panel-description">Forecasts remain available. Submit again to retry loading historical comparison data.</p>';}
  appendMatchupAnalysis({winner_prediction_id:winner?.prediction_id,rounds_prediction_id:rounds?.prediction_id,fighter_prediction_id:predictions[0]?.prediction_id,opponent_prediction_id:predictions[1]?.prediction_id},athletes,winner);

}
function aiGuidancePanel(guidance,athletes,winner) {
  if(!guidance)return '';
  const base=[guidance.baseline_p_fighter_win,guidance.baseline_p_opponent_win],adjusted=[guidance.p_fighter_win,guidance.p_opponent_win];
  const valid=guidance.status==='experimental'&&guidance.validated===false&&[...base,...adjusted].every(value=>Number.isFinite(value)&&value>=0&&value<=1)&&Math.abs(base[0]+base[1]-1)<1e-6&&Math.abs(adjusted[0]+adjusted[1]-1)<1e-6&&Math.abs(base[0]-winner?.p_fighter_win)<1e-6&&Math.abs(base[1]-winner?.p_opponent_win)<1e-6&&adjusted.every((value,index)=>Math.abs(value-base[index])<=.05000001)&&Number.isFinite(guidance.max_shift_percentage_points)&&guidance.max_shift_percentage_points<=5;
  if(!valid)return '<div class="ai-guidance"><h4>AI-guided estimate</h4><p class="panel-description">The experimental estimate is unavailable because its probability checks did not pass. The ML baseline remains above.</p></div>';
  const shift=(value,baseline)=>{const delta=(value-baseline)*100;return Math.abs(delta)<.00001?'No change':`${delta>0?'+':''}${number(delta,1)} percentage points`;};
  const explanation=typeof guidance.explanation==='string'?guidance.explanation:guidance.explanation?.text||'';
  return `<div class="ai-guidance" aria-label="Experimental AI-guided win estimate"><div class="winner-heading"><h4>AI-guided estimate</h4><span class="pill">EXPERIMENTAL · UNVALIDATED</span></div><div class="ai-guidance-sides">${athletes.map((athlete,index)=>`<div class="${index?'blue':'red'}-side"><strong>${escapeHtml(athlete.name)}</strong><div><span>${percent(base[index])}</span><span aria-hidden="true">→</span><b>${percent(adjusted[index])}</b></div><small>ML baseline → AI-guided · ${shift(adjusted[index],base[index])}</small></div>`).join('')}</div><p>${escapeHtml(explanation)}</p><p class="ai-guidance-warning">Experimental adjustment; backtest performance is not established. The language model can propose a bounded shift of up to ${number(guidance.max_shift_percentage_points,0)} percentage points. ML evaluation scores apply only to the unchanged baseline.</p><details class="winner-details"><summary>Adjustment policy</summary><p class="panel-description">Policy ${escapeHtml(guidance.policy_version)}. Among decisive outcomes. This adjustment is not a trained or calibrated probability model.</p></details></div>`;
}
function appendMatchupAnalysis(ids,athletes,winner) {
  const panel=document.createElement('section');panel.className='panel matchup-analysis';panel.setAttribute('aria-label','Matchup breakdown');panel.setAttribute('aria-live','polite');$('forecast-result').append(panel);
  const selection=matchupSelectionVersion;
  const heading='<p class="eyebrow">MATCHUP BREAKDOWN</p><h3>How this fight could unfold</h3>';
  const load=async()=>{
    panel.innerHTML=heading+'<div class="analysis-loading"><span class="loading-pulse"></span><p>Building a local breakdown from saved estimates and observed history. This can take up to 90 seconds; your forecasts remain available above.</p></div>';panel.setAttribute('aria-busy','true');
    if(!ids.winner_prediction_id){panel.innerHTML=heading+'<p class="panel-description">A saved winner forecast is required to build this breakdown. Compare the fighters again when that model is available.</p>';panel.setAttribute('aria-busy','false');return;}
    try{
      const report=await api('/api/v1/matchup-analysis',{method:'POST',body:JSON.stringify(Object.fromEntries(Object.entries(ids).filter(([,value])=>Boolean(value))))});
      if(!panel.isConnected||selection!==matchupSelectionVersion)return;
      if(report.status!=='ready')throw new Error('The matchup breakdown is not ready.');
      const claimText=claim=>typeof claim==='string'?claim:claim?.text||'';
      if(String(report.fighter_case?.fighter_id)!==String(athletes[0].fighter_id)||String(report.opponent_case?.fighter_id)!==String(athletes[1].fighter_id))throw new Error('The breakdown did not match the selected athletes.');
      const side=(item,athlete,color)=>`<article class="analysis-case ${color}"><h4>${escapeHtml(athlete.name)}</h4>${item?.advantages?.length?`<ul>${item.advantages.map(advantage=>`<li>${escapeHtml(advantage.text)}</li>`).join('')}</ul>`:''}<p>${escapeHtml(claimText(item?.path_to_victory)||'No supported path was returned.')}</p></article>`;
      panel.innerHTML=heading+`<p class="analysis-summary">${escapeHtml(claimText(report.summary))}</p>${aiGuidancePanel(report.ai_guidance,athletes,winner)}<div class="analysis-cases">${side(report.fighter_case,athletes[0],'red-side')}${side(report.opponent_case,athletes[1],'blue-side')}</div><div class="analysis-key"><span>Key stylistic question</span><p>${escapeHtml(claimText(report.key_matchup)||'No supported factor was returned.')}</p></div>${report.uncertainties?.length?`<div class="analysis-caveats"><h4>What could change the picture</h4><ul>${report.uncertainties.map(note=>`<li>${escapeHtml(note)}</li>`).join('')}</ul></div>`:''}<details class="winner-details"><summary>Sources & analysis details</summary><p class="panel-description">Generated by ${escapeHtml(report.provider)} / ${escapeHtml(report.model)} from saved model outputs. This is a model-based explanation, not certainty or betting advice.</p><dl class="metadata-list">${metadataRow('Analysis ID',report.analysis_id)}${metadataRow('Prompt version',report.prompt_version)}${metadataRow('Evidence SHA256',report.evidence_sha256)}${metadataRow('As of',report.as_of)}${metadataRow('Winner prediction',ids.winner_prediction_id)}${metadataRow('Rounds prediction',ids.rounds_prediction_id)}</dl>${report.evidence?.length?`<ul class="analysis-evidence">${report.evidence.map(row=>`<li><strong>${escapeHtml(row.label||row.id)}</strong> ${escapeHtml(row.text|| (typeof row.value==='object'?JSON.stringify(row.value):row.value))}</li>`).join('')}</ul>`:''}</details>`;
    }catch(error){if(!panel.isConnected||selection!==matchupSelectionVersion)return;panel.innerHTML=heading+`<p class="panel-description">The written breakdown is unavailable. Your forecasts remain available above.</p><button type="button" class="secondary-button analysis-retry">Retry breakdown ↗</button><details class="winner-details"><summary>Service details</summary><p class="panel-description">${escapeHtml(error.message)}</p></details>`;panel.querySelector('.analysis-retry').addEventListener('click',load);}
    finally{if(panel.isConnected)panel.setAttribute('aria-busy','false');}
  };load();
}
function renderObservedComparison(dashboard, window, athletes) {
  const pair=dashboard.windows?.[window]||{},a=pair.fighter||{},b=pair.opponent||{};
  const specs=[
    ['sig_landed_per_min','Landed / min','Higher = more output',2],
    ['sig_absorbed_per_min','Absorbed / min','Lower = fewer strikes absorbed',2],
    ['net_sig_per_min','Net strikes / min','Landed minus absorbed; higher = better differential',2],
    ['sig_attempted_per_min','Attempts / min','Higher = more attempts',2],
    ['accuracy','Accuracy','Landed / attempted',1,'percent'],
    ['defense','Defense','Share of opponent attempts avoided',1,'percent'],
    ['mean_duration_minutes','Mean duration','Minutes per covered bout',1],
    ['full_time_share','Full-duration share','Share reaching scheduled limit',1,'percent'],
  ];
  const rows=specs.map(([key,label,note,digits,kind])=>{
    const av=a[key],bv=b[key],valid=v=>v!==null&&v!==undefined&&Number.isFinite(Number(v));
    const scale=kind==='percent'?1:Math.max(Math.abs(Number(av)||0),Math.abs(Number(bv)||0),.01);
    const signed=key==='net_sig_per_min';
    const format=v=>valid(v)?(kind==='percent'?percent(v):`${signed&&v>0?'+':''}${number(v,digits)}`):'—';
    const bar=(v,side)=>{
      const width=valid(v)?Math.min(100,Math.abs(v)/scale*100):0;
      if(signed)return `<div class="metric-track signed"><span class="zero-marker"></span><i class="${side}" style="left:${v<0?50-width/2:50}%;width:${width/2}%"></i></div>`;
      return `<div class="metric-track ${side}"><i style="width:${width}%"></i></div>`;
    };
    return `<div class="mirrored-metric"><div class="metric-value red-side">${format(av)}</div><div class="metric-name"><strong>${escapeHtml(label)}</strong><small>${escapeHtml(note)}</small></div><div class="metric-value blue-side">${format(bv)}</div><div class="left-track">${bar(av,'red-side')}</div><div class="metric-center">${signed?'− ← 0 → +':kind==='percent'?'0–100%':'Shared scale'}</div><div>${bar(bv,'blue-side')}</div></div>`;
  }).join('');
  const history=(items,name,side)=>{
    const rows=(window==='last5'?items.slice(0,5):items);
    return `<div class="history-column ${side}"><h4>${escapeHtml(name)}</h4>${rows.length?`<div class="table-scroll"><table><thead><tr><th>DATE / OPPONENT</th><th>LANDED</th><th>ABS.</th><th>MIN.</th></tr></thead><tbody>${rows.map(row=>`<tr><td>${escapeHtml(date(row.event_date))}<small>${escapeHtml(row.opponent_name)}</small><small>${escapeHtml(row.method||'')} · ${escapeHtml(row.scheduled_rounds)} rounds</small></td><td>${number(row.sig_landed,0)}</td><td>${number(row.sig_absorbed,0)}</td><td>${number(row.duration_minutes,1)}</td></tr>`).join('')}</tbody></table></div>`:'<p class="panel-description">No covered prior bouts. Rates are unavailable.</p>'}</div>`;
  };
  return `<div class="observed-heading"><div><p class="eyebrow">OBSERVED HISTORY · NOT A PROJECTION</p><h3>Striking, side by side</h3></div><div class="history-window" role="group" aria-label="Historical sample"><button type="button" data-history-window="last5" aria-pressed="${window==='last5'}">Last 5</button><button type="button" data-history-window="career" aria-pressed="${window==='career'}">All covered</button></div></div><p class="panel-description">${escapeHtml(dashboard.coverage_label||'Covered UFC bouts since 2013')} · History before ${escapeHtml(date(dashboard.as_of))}. Rates aggregate observed counts and time; these are not full MMA career records.</p><div class="sample-comparison"><span class="red-side"><b>${number(a.covered_bouts,0)}</b> bouts · ${number(a.observed_minutes,1)} observed min</span><span class="blue-side"><b>${number(b.covered_bouts,0)}</b> bouts · ${number(b.observed_minutes,1)} observed min</span></div>${rows}<p class="panel-description comparison-scale-note">Each row uses the same scale for both athletes. Net bars share a zero center; left of zero means more absorbed than landed. No bar alone determines matchup advantage.</p><p class="panel-description">${escapeHtml(dashboard.note||'Historical opponent quality, styles, and sample sizes can differ. These summaries are descriptive.')}</p>`;
}
function distributionChart(prediction) {
  let points = (prediction.distribution || [])
    .map((p) => ({ x: Number(p.count), p: Number(p.probability) }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.p) && p.p >= 0)
    .sort((a, b) => a.x - b.x);
  if (!points.length)
    return '<p class="panel-description">Distribution detail is unavailable for this artifact.</p>';
  const width = 650,
    height = 205,
    left = 28,
    right = 12,
    top = 18,
    bottom = 34;
  const max = Math.max(...points.map((p) => p.p), 0.001);
  const upper = Math.max(
    points.at(-1).x,
    Number(prediction.line) * 1.05,
    Number(prediction.interval_high) * 1.05,
    10,
  );
  const x = (v) => left + (v / upper) * (width - left - right),
    y = (v) => height - bottom - (v / max) * (height - top - bottom);
  const groups = [];
  const stride = Math.max(1, Math.ceil(points.length / 130));
  for (let i = 0; i < points.length; i += stride) {
    const group = points.slice(i, i + stride);
    groups.push({
      x: group.reduce((sum, p) => sum + p.x, 0) / group.length,
      p: group.reduce((sum, p) => sum + p.p, 0) / group.length,
    });
  }
  const barWidth = Math.max(
    1,
    ((width - left - right) / Math.max(groups.length, 1)) * 0.83,
  );
  const bars = groups
    .map(
      (p) =>
        `<rect x="${x(p.x) - barWidth / 2}" y="${y(p.p)}" width="${barWidth}" height="${height - bottom - y(p.p)}" fill="${p.x > prediction.line ? "#d4ec89" : "#52616b"}" opacity=".88"/>`,
    )
    .join("");
  const ticks = Array.from({ length: 6 }, (_, i) => (upper * i) / 5);
  return `<svg class="distribution-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Probability distribution of significant strikes. Line ${escapeHtml(prediction.line)}; interval ${escapeHtml(prediction.interval_low)} to ${escapeHtml(prediction.interval_high)}."><rect x="${x(prediction.interval_low)}" y="${top}" width="${Math.max(0, x(prediction.interval_high) - x(prediction.interval_low))}" height="${height - top - bottom}" fill="#d4ec89" opacity=".04"/><line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#46535a"/>${bars}<line x1="${x(prediction.line)}" y1="${top - 5}" x2="${x(prediction.line)}" y2="${height - bottom}" stroke="#eca771" stroke-dasharray="4 3"/><text x="${Math.min(width - 70, Math.max(left, x(prediction.line) + 6))}" y="12" fill="#eca771" font-size="9">LINE ${escapeHtml(prediction.line)}</text>${ticks.map((t) => `<text x="${x(t)}" y="${height - 15}" text-anchor="middle" fill="#a2adb3" font-size="9">${Math.round(t)}</text>`).join("")}<line x1="${x(prediction.interval_low)}" y1="${height - 3}" x2="${x(prediction.interval_high)}" y2="${height - 3}" stroke="#d4ec89" stroke-width="2"/></svg>`;
}
function renderPrediction(p) {
  const name = p.fighter?.name || "Selected athlete",
    opponent = p.opponent?.name || "Opponent";
  $("forecast-result").innerHTML =
    `<div class="panel result-panel"><div class="result-heading"><div><p class="eyebrow">SIGNIFICANT STRIKES LANDED</p><h2>${escapeHtml(name)}</h2><p class="result-subtitle">vs ${escapeHtml(opponent)} · ${escapeHtml(p.scheduled_rounds)} scheduled rounds</p></div><span class="pill">MODEL FORECAST</span></div><div class="forecast-numbers"><div><span class="stat-label">EXPECTED STRIKES</span><strong class="big-stat">${number(p.mean)}</strong><span class="stat-footnote">Median ${number(p.median, 0)} strikes</span></div><div><span class="stat-label">${number((p.interval_level ?? 0.8) * 100, 0)}% PREDICTION INTERVAL</span><strong class="big-stat interval">${number(p.interval_low, 0)}–${number(p.interval_high, 0)}</strong><span class="stat-footnote">Range of possible fight outcomes</span></div></div><div class="chart-wrap"><p class="chart-title">COUNT DISTRIBUTION / STRIKES LANDED</p>${distributionChart(p)}</div><div class="chart-legend"><span><i class="legend-dot"></i>Below line</span><span><i class="legend-dot over"></i>Above line</span><span><i class="legend-line"></i>Your line</span></div><div class="probability-grid"><div class="probability-card"><span>OVER ${escapeHtml(p.line)}</span><strong>${percent(p.p_over)}</strong></div><div class="probability-card under"><span>UNDER ${escapeHtml(p.line)}</span><strong>${percent(p.p_under)}</strong></div></div><p class="push-note">Exact-line probability: ${percent(p.p_push)}${Number.isInteger(p.line) ? " · Integer lines can push." : " · Half-point lines cannot push."}</p>${p.warnings?.length ? `<ul class="warnings">${p.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>` : ""}<div class="provenance">MODEL ${escapeHtml(p.model_version)} · FEATURES THROUGH ${escapeHtml(date(p.feature_cutoff))}<br>Data through ${escapeHtml(date(p.data_cutoff))} · History: ${escapeHtml(p.history?.fighter_bouts ?? "—")} athlete / ${escapeHtml(p.history?.opponent_bouts ?? "—")} opponent bouts</div></div>`;
}
async function loadRecent() {
  if (state.loaded.has("recent")) return;
  try {
    const data = await api("/api/v1/recent");
    state.recent = data;
    state.loaded.add("recent");
    $("recent-period").textContent =
      `${date(data.from || "2026-10-08")} → ${date(data.to || "2026-10-22")}`;
    const rows = data.predictions || [],
      m = metricValues(data);
    if (!rows.length) {
      $("recent-content").innerHTML =
        '<div class="panel data-panel"><h3>No eligible fights in this window.</h3><p class="panel-description">The replay is empty because no eligible completed bouts were found. No results have been substituted from another period.</p></div>';
      return;
    }
    $("recent-content").innerHTML =
      `<div class="metrics-grid">${metric("ATHLETE FORECASTS", number(m.n ?? rows.length, 0), "Two athletes can share one bout")}${metric("MEAN ABSOLUTE ERROR", number(m.mae), "Strikes per athlete")}${metric("ROOT MEAN SQUARE ERROR", number(m.rmse), "Larger misses carry more weight")}${metric(`${number(m.level * 100, 0)}% INTERVAL COVERAGE`, percent(m.coverage), "Observed share inside the interval")}</div><div class="panel data-panel"><h3>Forecasts meet the scorecard</h3><p class="panel-description">Each dot is one athlete forecast. The diagonal marks a perfect count prediction.</p>${replayChart(rows)}</div><div class="panel data-panel"><div class="panel-heading"><h3>Fight-by-fight replay</h3><span class="small-label">SYNTHETIC LINES · NO ROI CLAIM</span></div><div class="table-scroll"><table><thead><tr><th>ATHLETE / OPPONENT</th><th>DATE</th><th>EXPECTED</th><th>ACTUAL</th><th>INTERVAL</th><th>ABS. ERROR</th><th>LINE</th><th>P(OVER)</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${escapeHtml(r.fighter_name)}<small>vs ${escapeHtml(r.opponent_name)}</small></td><td>${escapeHtml(date(r.event_date))}</td><td>${number(r.mean)}</td><td class="positive">${number(r.actual, 0)}</td><td>${number(r.interval_low, 0)}–${number(r.interval_high, 0)}</td><td>${number(r.absolute_error ?? Math.abs(r.actual - r.mean))}</td><td>${number(r.line)}</td><td>${percent(r.p_over)}</td></tr>`).join("")}</tbody></table></div></div><p class="section-copy">${escapeHtml(data.description || data.warning || "Retrospective replay using pre-window training and pre-bout features. Lines are synthetic; outcomes are observed historical counts.")}</p>`;
  } catch (error) {
    $("recent-content").innerHTML = errorPanel(error.message);
  }
}
function replayChart(rows) {
  const points = rows.filter(
    (r) => Number.isFinite(Number(r.actual)) && Number.isFinite(Number(r.mean)),
  );
  if (!points.length) return "";
  const w = 850,
    h = 260,
    l = 48,
    r = 25,
    t = 15,
    b = 40,
    max =
      Math.ceil(
        Math.max(
          ...points.flatMap((p) => [Number(p.mean), Number(p.actual)]),
          20,
        ) / 20,
      ) * 20;
  const x = (v) => l + (v / max) * (w - l - r),
    y = (v) => h - b - (v / max) * (h - t - b);
  const ticks = Array.from({ length: 5 }, (_, i) => (max * i) / 4);
  return `<svg class="replay-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Scatter plot of predicted versus actual significant strikes for ${points.length} athletes">${ticks.map((v) => `<line x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}" stroke="#35424a"/><text x="${l - 10}" y="${y(v) + 3}" fill="#a2adb3" text-anchor="end" font-size="9">${Math.round(v)}</text><text x="${x(v)}" y="${h - 23}" fill="#a2adb3" text-anchor="middle" font-size="9">${Math.round(v)}</text>`).join("")}<line x1="${x(0)}" y1="${y(0)}" x2="${x(max)}" y2="${y(max)}" stroke="#a2adb3" stroke-dasharray="4 5"/>${points.map((p) => `<circle cx="${x(Number(p.mean))}" cy="${y(Number(p.actual))}" r="4" fill="#d4ec89" fill-opacity=".72"><title>${escapeHtml(p.fighter_name)}: expected ${number(p.mean)}, actual ${number(p.actual, 0)}</title></circle>`).join("")}<text x="${w / 2}" y="${h - 2}" fill="#a2adb3" font-size="9" text-anchor="middle">EXPECTED STRIKES</text><text transform="translate(10 ${h / 2}) rotate(-90)" fill="#a2adb3" font-size="9" text-anchor="middle">ACTUAL STRIKES</text></svg>`;
}
function metadataRow(label, value) {
  return `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value ?? "—")}</dd></div>`;
}
async function loadEvidence() {
  if (state.loaded.has("evidence")) return;
  const results = await Promise.allSettled([
    api("/api/v1/evaluation"),
    api("/api/v1/models/current"),
    api("/api/v1/benchmark"),
  ]);
  if (results[0].status === "rejected" && results[1].status === "rejected") {
    $("evidence-content").innerHTML = errorPanel(results[0].reason.message);
    return;
  }
  state.loaded.add("evidence");
  const report = results[0].status === "fulfilled" ? results[0].value : {},
    model = results[1].status === "fulfilled" ? results[1].value : {},
    benchmark = results[2].status === "fulfilled" ? results[2].value : null;
  const m = metricValues(report),
    baseline = report.baseline || report.baseline_metrics || {},
    features =
      model.feature_names || model.features || report.feature_names || [],
    limitations = model.limitations ||
      report.limitations || [
        "Historical performance does not guarantee future accuracy.",
        "Sparse athlete histories and changing styles can increase forecast uncertainty.",
        "Synthetic strike lines measure model behavior; they do not establish profitability.",
      ];
  const splits = report.splits || [];
  $("evidence-content").innerHTML =
`<p class="section-copy">Historical report: ${escapeHtml((report.report_period || []).join(" → ") || "See evaluation splits below")}. These metrics evaluate time-separated historical artifacts. The serving artifact is separately versioned; inspect its fitting and calibration cutoffs below.</p><div class="metrics-grid">${metric("HISTORICAL FORECASTS", number(m.n, 0), "Chronological evaluation sample")}${metric("MODEL MAE", number(m.mae, 2), "Strikes per athlete")}${metric("BASELINE MAE", number(baseline.mae ?? baseline.mean_absolute_error, 2), "Compare on the same historical sample")}${metric(`${number(m.level * 100, 0)}% INTERVAL COVERAGE`, percent(m.coverage), "Empirical historical coverage")}</div><div class="evidence-columns"><div class="panel data-panel"><h3>Artifact & provenance</h3><dl class="metadata-list">${metadataRow("Model version", model.model_version || model.current_version || model.version || report.model_version)}${metadataRow("Model family", model.model_type || model.estimator || model.algorithm || model.champion)}${metadataRow("Training cutoff", date(model.training_cutoff || model.train_cutoff || model.data_through || report.training_cutoff))}${metadataRow("Calibration cutoff", date(model.calibration_cutoff))}${metadataRow("Serving channel", model.serving_channel)}${metadataRow("Data cutoff", date(model.data_cutoff || model.data_through))}${metadataRow("Built at", model.trained_at || model.created_at)}${metadataRow("Artifact SHA256", (typeof model.artifact_sha256 === "object" ? model.artifact_sha256?.current : model.artifact_sha256) || model.sha256 || model.artifact_hash)}${metadataRow("Feature schema", model.feature_schema_version || model.feature_version)}</dl></div><div class="panel data-panel"><h3>How to read these results</h3><ul class="limitation-list">${(Array.isArray(limitations) ? limitations : [limitations]).map((l) => `<li>${escapeHtml(l)}</li>`).join("")}</ul><p class="panel-description">MAE is the average absolute count error. Coverage is the fraction of observed outcomes inside the stated interval; compare it with the interval’s nominal level.</p></div></div>${evidenceDiagnostics(report)}${splits.length ? `<div class="panel data-panel"><h3>Time-based evaluation splits</h3><div class="table-scroll"><table><thead><tr><th>SPLIT</th><th>START</th><th>END</th><th>SAMPLES</th></tr></thead><tbody>${splits.map((s) => `<tr><td>${escapeHtml(s.name)}</td><td>${escapeHtml(date(s.start))}</td><td>${escapeHtml(date(s.end))}</td><td>${number(s.n ?? s.count, 0)}</td></tr>`).join("")}</tbody></table></div></div>` : ""}${features.length ? `<div class="panel data-panel"><h3>Inputs available before the fight</h3><p class="panel-description">Feature values are reconstructed using only eligible history before each forecast cutoff.</p><div class="feature-tags">${features.map((f) => `<span class="feature-tag">${escapeHtml(typeof f === "string" ? f : f.name)}</span>`).join("")}</div></div>` : ""}${benchmark && benchmark.status !== "not_measured" ? renderBenchmark(benchmark) : '<div class="panel data-panel"><h3>Runtime benchmark</h3><p class="panel-description">No benchmark artifact is available. Run the benchmark command to record measured latency and throughput.</p></div>'}<div class="panel data-panel"><h3>Inspect the service</h3><p class="panel-description">The same documented endpoints power this interface. Request schemas, validation rules, and response fields are available in the interactive OpenAPI reference.</p><a class="text-link" href="/docs">Open API documentation ↗</a></div>`;
}
function renderBenchmark(b) {
  return `<div class="panel data-panel"><h3>Measured serving performance</h3><dl class="metadata-list">${metadataRow("Requests", b.requests ?? b.n ?? b.sample_count)}${metadataRow("Concurrency", b.concurrency)}${metadataRow("p50 latency", b.p50_ms !== undefined ? `${number(b.p50_ms)} ms` : undefined)}${metadataRow("p95 latency", b.p95_ms !== undefined ? `${number(b.p95_ms)} ms` : undefined)}${metadataRow("Throughput", b.requests_per_second !== undefined ? `${number(b.requests_per_second)} requests/s` : undefined)}${metadataRow("Error rate", percent(b.error_rate))}${metadataRow("Measured at", b.measured_at)}${metadataRow("Environment", b.environment || b.hardware)}</dl><p class="panel-description">${escapeHtml(b.workload || "Performance is specific to the recorded environment and workload.")} ${escapeHtml(b.notes || "")}</p></div>`;
}
function evidenceDiagnostics(report) {
  const comparisons = report.metrics || {},
    groups = report.subgroups || {},
    calibration = report.calibration || [];
  let html = "";
  if (
    Object.values(comparisons).some(
      (m) => m && typeof m === "object" && m.mae !== undefined,
    )
  )
    html += `<div class="panel data-panel"><h3>Same sample. Different models.</h3><p class="panel-description">Held-out count errors and distribution quality. Lower MAE and negative log likelihood are better; coverage should approach 80%.</p><div class="table-scroll"><table><thead><tr><th>MODEL</th><th>SAMPLES</th><th>MAE</th><th>RMSE</th><th>NEG. LOG LIKELIHOOD</th><th>80% COVERAGE</th></tr></thead><tbody>${Object.entries(
      comparisons,
    )
      .filter(([, m]) => m && typeof m === "object" && m.mae !== undefined)
      .map(
        ([name, m]) =>
          `<tr><td>${escapeHtml(name)}${name === report.champion ? " · selected" : ""}</td><td>${number(m.n, 0)}</td><td>${number(m.mae, 2)}</td><td>${number(m.rmse, 2)}</td><td>${number(m.nll, 3)}</td><td>${percent(m.interval_coverage ?? m.coverage_80)}</td></tr>`,
      )
      .join("")}</tbody></table></div></div>`;
  if (calibration.length) {
    const w = 480,
      h = 260,
      l = 42,
      r = 20,
      t = 15,
      b = 35,
      x = (v) => l + v * (w - l - r),
      y = (v) => h - b - v * (h - t - b);
    html += `<div class="evidence-columns"><div class="panel data-panel"><h3>Probability calibration</h3><p class="panel-description">Synthetic-line over probabilities grouped into bins. Larger dots contain more forecasts. The diagonal marks agreement.</p><svg class="replay-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Predicted over probability versus observed frequency">${[0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="${l}" x2="${w - r}" y1="${y(v)}" y2="${y(v)}" stroke="#35424a"/><text x="${l - 8}" y="${y(v) + 3}" text-anchor="end" fill="#a2adb3" font-size="9">${Math.round(v * 100)}%</text><text x="${x(v)}" y="${h - 18}" text-anchor="middle" fill="#a2adb3" font-size="9">${Math.round(v * 100)}%</text>`).join("")}<line x1="${x(0)}" x2="${x(1)}" y1="${y(0)}" y2="${y(1)}" stroke="#a2adb3" stroke-dasharray="4 4"/>${calibration.map((c) => `<circle cx="${x(c.predicted)}" cy="${y(c.observed)}" r="${Math.min(12, 3 + Math.sqrt(c.n) / 3)}" fill="#d4ec89" fill-opacity=".75"><title>Predicted ${percent(c.predicted)}, observed ${percent(c.observed)}, n=${c.n}</title></circle>`).join("")}<text x="${w / 2}" y="${h - 1}" text-anchor="middle" fill="#a2adb3" font-size="9">PREDICTED PROBABILITY</text></svg></div><div class="panel data-panel"><h3>Where forecasts are harder</h3><p class="panel-description">Subgroup outcomes on the historical report sample. Smaller groups are less stable.</p><div class="table-scroll"><table><thead><tr><th>GROUP</th><th>N</th><th>MAE</th><th>COVERAGE</th></tr></thead><tbody>${Object.entries(
      groups,
    )
      .map(
        ([name, m]) =>
          `<tr><td>${escapeHtml(name.replaceAll("_", " "))}</td><td>${number(m.n, 0)}</td><td>${number(m.mae)}</td><td>${percent(m.coverage_80)}</td></tr>`,
      )
      .join("")}</tbody></table></div></div></div>`;
  }
  return html;
}
for (const id of ["fighter", "opponent"]) {
  $(`${id}-search`).addEventListener("input", (event) => {
    const selection = $(id).value,
      other = $(id === "fighter" ? "opponent" : "fighter").value,
      query = event.target.value.trim().toLowerCase();
    const filtered = state.fighters.filter(
      (f) =>
        f.name.toLowerCase().includes(query) ||
        String(f.fighter_id) === selection,
    );
    $(id).innerHTML = filtered
      .map(
        (f) =>
          `<option value="${escapeHtml(f.fighter_id)}" ${String(f.fighter_id) === other ? "disabled" : ""}>${escapeHtml(f.name)}${f.bout_count !== undefined ? ` · ${f.bout_count} bouts` : ""}</option>`,
      )
      .join("");
    $(id).value = selection;
  });
}
function safeUrl(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  try {
    const parsed = new URL(value, location.origin);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : null;
  } catch { return null; }
}
function factText(value) {
  if (value === null || value === undefined) return "Unavailable";
  if (typeof value === "object") return value.description || value.message || value.status || JSON.stringify(value);
  return String(value);
}
function addMatchupDisclosure(prediction) {
  const details = document.createElement("details");
  details.className = "matchup-disclosure panel";
  details.innerHTML = '<summary>Why this projection? <span class="text-link">Explore history + sensitivity</span></summary><div class="matchup-context"><p class="panel-description">Open to load pre-fight history and model sensitivity. This separate request does not affect the forecast endpoint benchmark.</p></div>';
  $("forecast-result").append(details);
  let loaded = false;
  details.addEventListener("toggle", async () => {
    if (!details.open || loaded) return;
    const container = details.querySelector(".matchup-context");
    container.innerHTML = '<div class="loading-state">Loading matchup context…</div>';
    try {
      const params = new URLSearchParams({fighter_id:prediction.fighter.id, opponent_id:prediction.opponent.id, scheduled_rounds:String(prediction.scheduled_rounds)});
      const context = await api(`/api/v1/matchups?${params}`);
      container.innerHTML = renderMatchupContext(context, prediction);
      loaded = true;
    } catch (error) { container.innerHTML = `<p class="error-message">${escapeHtml(error.message)}</p><p class="panel-description">Close and reopen this panel to retry.</p>`; }
  });
}
function renderMatchupContext(context, prediction) {
  const rows = context.comparison || [];
  const explanations = context.explanation || [];
  const history = (items, name) => `<div><h3>${escapeHtml(name)} · last ${items.length}</h3>${items.length ? `<div class="history-spark">${items.map(item => `<div><span style="height:${Math.max(3,Math.min(100,Number(item.strikes)/Math.max(...items.map(i=>Number(i.strikes)),1)*100))}%" title="${escapeHtml(date(item.event_date))}: ${escapeHtml(item.strikes)} strikes"></span><small>${number(item.strikes,0)}</small></div>`).join("")}</div><table><thead><tr><th>DATE / OPPONENT</th><th>LANDED</th></tr></thead><tbody>${items.map(item=>`<tr><td>${escapeHtml(date(item.event_date))}<small>${escapeHtml(item.opponent||item.opponent_name)}</small></td><td>${number(item.strikes,0)}</td></tr>`).join("")}</tbody></table>` : '<p class="panel-description">No eligible prior bouts.</p>'}</div>`;
  return `<p class="panel-description">Historical summaries provide context; they do not make this hypothetical matchup a scheduled fixture.</p><div class="history-grid">${history(context.fighter_recent||[],prediction.fighter.name)}${history(context.opponent_recent||[],prediction.opponent.name)}</div>${rows.length?`<div class="table-scroll"><table><thead><tr><th>PRE-FIGHT COMPARISON</th><th>${escapeHtml(prediction.fighter.name)}</th><th>${escapeHtml(prediction.opponent.name)}</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${escapeHtml(r.label)}<small>${escapeHtml(r.unit||"")}</small></td><td>${number(r.fighter,2)}</td><td>${number(r.opponent,2)}</td></tr>`).join("")}</tbody></table></div>`:""}${explanations.length?`<h3 class="sensitivity-heading">Model sensitivity, not causal attribution</h3><p class="panel-description">${escapeHtml(context.explanation_note||"Changes in predicted mean under controlled feature substitutions. These are model responses, not causal effects or a decomposition that must sum to the forecast.")}</p><div class="sensitivity-list">${explanations.map(e=>`<div><span>${escapeHtml(e.group)}</span><strong>${Number(e.delta_mean)>0?"+":""}${number(e.delta_mean)} strikes</strong><small>${escapeHtml(e.description)}</small></div>`).join("")}</div>`:""}`;
}
function forwardModelComparison(summary) {
  const rows=Object.entries(summary.by_model||{});
  if(!rows.length)return '';
  return `<div class="panel data-panel"><h3>Issued forecasts by model</h3><p class="panel-description">${number(summary.n_issued??summary.issued,0)} saved forecast records cover ${number(summary.unique_fixture_athlete_count,0)} unique fixture/athlete perspectives. Model versions can forecast the same outcomes; these are not independent observations.</p><div class="table-scroll"><table><thead><tr><th>MODEL</th><th>ISSUED</th><th>SETTLED</th><th>MAE</th><th>NLL</th></tr></thead><tbody>${rows.map(([version,m])=>`<tr><td title="${escapeHtml(version)}">${escapeHtml(version.startsWith('v3-')?'V3 duration mixture':version.startsWith('v2-')?'V2 baseline':version)}<small>${escapeHtml(version)}</small></td><td>${number(m.n_issued??m.issued,0)}</td><td>${number(m.n_settled??m.scored_count,0)}</td><td>${number(m.mae,2)}</td><td>${m.nll_status==='infinite'?'∞':number(m.nll,4)}</td></tr>`).join('')}</tbody></table></div><p class="panel-description">Scores remain unavailable until matched outcomes settle. NLL uses validated frozen distributions only; inspect the export for scoring coverage and missing legacy parameters.</p></div>`;
}
async function loadForward(force=false) {
  if (state.loaded.has("forward") && !force) return;
  try {
    const report = await api("/api/v1/forward");
    state.loaded.add("forward");
    const summary=report.summary||{}, batches=report.batches||[], upcoming=report.upcoming||{};
    const pending=summary.pending_count??summary.n_pending??summary.pending, scored=summary.scored_count??summary.n_settled??summary.scored;
$("forward-content").innerHTML=`<div class="metrics-grid">${metric("FROZEN BATCHES",number(summary.batch_count??batches.length,0),"Persisted before scheduled events")}${metric("PENDING FORECASTS",number(pending,0),"Versioned records; outcomes can repeat")}${metric("SCORED FORECASTS",number(scored,0),"Observed results linked later")}${metric("FORWARD MAE",number(summary.mae,2),"Unavailable until outcomes are scored")}</div>${batches.length?`<div class="panel data-panel"><h3>Immutable forecast batches</h3><div class="table-scroll"><table><thead><tr><th>BATCH</th><th>FROZEN AT</th><th>FIXTURES</th><th>SCORED</th><th>STATUS</th></tr></thead><tbody>${batches.map(batch=>`<tr><td>${escapeHtml(batch.title||batch.batch_id)}<small>${escapeHtml(batch.batch_id||batch.id)}</small></td><td>${escapeHtml(batch.created_at||batch.recorded_at||batch.frozen_at)}</td><td>${number(batch.fixture_count??batch.n_entries??batch.prediction_count??batch.n,0)}</td><td>${number(batch.scored_count??batch.n_settled,0)}</td><td>${escapeHtml(batch.status||batch.mode||"Recorded")}</td></tr>`).join("")}</tbody></table></div></div>`:`<div class="panel forward-empty"><span class="empty-symbol">↗</span><div><p class="eyebrow">NO FORWARD EVIDENCE YET</p><h3>The clean ledger is the starting point.</h3><p>Verify a future fixture, choose a line and record its source, then freeze the batch below. Results appear only after an observed outcome is linked. The recent replay has not been inserted here.</p></div></div>`}<div class="forward-guide"><span class="outlined-label">RECORD → WAIT → SCORE</span><p>${escapeHtml(factText(report.guide?.description||report.guide?.message||"Scheduled timestamps must be in the future when recorded. Predictions and source labels are frozen; outcomes can be attached later without rewriting the forecast."))}</p></div>${upcoming.fixtures?.length?`<div class="panel data-panel"><h3>Upcoming fixture references</h3><p class="panel-description">Verify scheduled time and athlete identities before creating a batch. Availability does not imply a line feed.</p><div class="table-scroll"><table><thead><tr><th>FIXTURE</th><th>SCHEDULED</th><th>REFERENCE</th></tr></thead><tbody>${upcoming.fixtures.map(f=>`<tr><td>${escapeHtml(f.fighter_name||f.fighter_id)} vs ${escapeHtml(f.opponent_name||f.opponent_id)}</td><td>${escapeHtml(f.scheduled_at||f.event_date)}</td><td>${safeUrl(f.source_url)?`<a href="${escapeHtml(safeUrl(f.source_url))}" target="_blank" rel="noopener">Source ↗</a>`:"Verify source"}</td></tr>`).join("")}</tbody></table></div></div>`:""}`;
    $('forward-content').querySelector('.metrics-grid')?.insertAdjacentHTML('afterend',forwardModelComparison(summary));
  } catch (error) { $("forward-content").innerHTML=errorPanel(error.message); }
}
$("forward-use-scenario").addEventListener("click",()=>{
  const message=$("forward-message"), scheduled=$("forward-scheduled").value;
  if(!scheduled || !$("forward-title").value.trim() || !$("forward-line-source").value.trim()) { message.textContent="Enter a batch title, future schedule, and a public schedule URL before building the request.";return; }
  if(!$("fighter").value || !$("opponent").value || !state.fighters.length){message.textContent="Load a valid matchup in Forecast Explorer first.";return;}
  const time=new Date(scheduled);
  if(!Number.isFinite(time.getTime())||time<=new Date()){message.textContent="The scheduled time must be in the future. Historical replays belong in Recent Cards.";return;}
  const source=$("forward-line-source").value.trim();
  if(!/^https?:\/\//i.test(source)||!safeUrl(source)){message.textContent="Provide an HTTP or HTTPS public schedule URL.";return;}
  const identity=crypto.randomUUID();
  const request={title:$("forward-title").value.trim(),idempotency_key:identity,fixtures:[{fixture_id:`manual-${identity}`,scheduled_at:time.toISOString(),event_date:scheduled.slice(0,10),schedule_kind:"user_attested",source_url:source,fighter_id:$("fighter").value,opponent_id:$("opponent").value,scheduled_rounds:Number(document.querySelector('[name=scheduled_rounds]:checked').value),line:Number($("line").value),line_kind:$("forward-line-kind").value}]};
  $("forward-json").value=JSON.stringify(request,null,2);
  message.textContent="JSON prepared from the current explorer scenario. Verify the fixture and line provenance before freezing this batch.";
});
$("forward-file").addEventListener("change",async event=>{
  const file=event.target.files?.[0];if(!file)return;
  if(file.size>1000000){$("forward-message").textContent="Use a JSON file smaller than 1 MB.";return;}
  try {const parsed=JSON.parse(await file.text());$("forward-json").value=JSON.stringify(parsed,null,2);$("forward-message").textContent="File imported locally. Review the request before submitting.";}
  catch {$("forward-message").textContent="This file is not valid JSON.";}
});
$("forward-submit").addEventListener("click",async()=>{
  const button=$("forward-submit"),message=$("forward-message");
  try {
    const request=JSON.parse($("forward-json").value);
    if(!request.title?.trim()||!request.idempotency_key?.trim()||!Array.isArray(request.fixtures)||!request.fixtures.length)throw new Error("Provide a title, idempotency key, and at least one verified future fixture.");
    button.disabled=true;message.textContent="Freezing batch and model provenance…";
    const response=await api('/api/v1/forward/batches',{method:'POST',body:JSON.stringify(request)});
    message.textContent=`Batch recorded: ${response.batch_id||response.id||request.title}. Keep the idempotency key for safe retries. No outcome has been inferred from this submission.`;
    await loadForward(true);
  } catch(error){message.textContent=error.message;}
  finally{button.disabled=false;}
});
async function loadExperiments() {
  const details=document.createElement('details');details.className='panel data-panel report-details';
  const summary=document.createElement('summary');summary.textContent='Earlier feature study (V2)';details.append(summary);
  const container=document.createElement('div');container.id='experiments-content';container.innerHTML='<p class="panel-description">Loading the saved experiment report…</p>';
  details.append(container);$('evidence-content').append(details);
  try {
    const report=await api('/api/v1/experiments');
    const rows=report.development_aggregate||report.experiments||report.candidates||report.results||[];
    const list=Array.isArray(rows)?rows:Object.entries(rows).map(([name,result])=>({name,...result}));
    container.innerHTML=`<h3>Challenger experiments</h3><p class="panel-description">${escapeHtml(factText(report.selection_decision||report.decision||report.summary||'Saved research runs. A candidate is not promoted simply because one metric improves.'))}</p><p class="panel-description">Selection sample: development years ${escapeHtml((report.selection_period||[]).join(" → ")||"see report")}. Lower negative log likelihood is the declared selection rule.</p>${list.length?`<div class="table-scroll"><table><thead><tr><th>CANDIDATE</th><th>SAMPLES</th><th>MAE</th><th>NLL</th><th>80% COVERAGE</th><th>STATUS</th></tr></thead><tbody>${list.map(row=>{const m=row.metrics||row.test||row;return `<tr><td>${escapeHtml(row.name||row.model||row.candidate)}<small>${escapeHtml(row.notes||row.description||'')}</small></td><td>${number(m.n??((report.development||[]).filter(d=>d.candidate===row.name).reduce((sum,d)=>sum+d.n,0)||null),0)}</td><td>${number(m.mae,2)}</td><td>${number(m.nll,3)}</td><td>${percent(m.coverage_80??m.interval_coverage)}</td><td>${escapeHtml(row.status||(row.name===report.selected_candidate?'Selected family':'Not promoted'))}</td></tr>`;}).join('')}</tbody></table></div>`:'<p class="panel-description">No completed candidate rows are present. The current production artifact is unchanged.</p>'}${report.retrospective?`<p class="error-message">${escapeHtml(report.retrospective_comparison_label||"The additional historical comparison reuses already-seen outcomes and is not a fresh holdout.")}</p>`:""}${report.warnings?.length?`<ul class="limitation-list">${report.warnings.map(w=>`<li>${escapeHtml(w)}</li>`).join("")}</ul>`:""}<details class="report-details"><summary>Inspect exact experiment report</summary><pre>${escapeHtml(JSON.stringify(report,null,2))}</pre></details>`;
  } catch(error){container.innerHTML=`<h3>Challenger experiments</h3><p class="panel-description">No saved experiment report is available: ${escapeHtml(error.message)}. Historical evidence above describes the current evaluated model family.</p>`;}
}
function studyRows(entries, selected) {
  return entries.map(([name, metrics]) => {
    const m=metrics.metrics||metrics;
    return `<tr><td>${escapeHtml(name.replaceAll('_',' '))}${name===selected?' <small>Selected on development years</small>':''}</td><td>${number(m.n,0)}</td><td>${number(m.nll,4)}</td><td>${number(m.mae,2)}</td><td>${percent(m.coverage_80)}</td><td>${number(m.interval_width_80,2)}</td></tr>`;
  }).join('');
}
function studyTable(entries, selected) {
  return `<div class="table-scroll"><table><thead><tr><th>CANDIDATE</th><th>FORECASTS</th><th>NLL ↓</th><th>MAE ↓</th><th>80% COVERAGE</th><th>80% WIDTH</th></tr></thead><tbody>${studyRows(entries,selected)}</tbody></table></div>`;
}
function studyDifference(label, result) {
  if(!result)return '';
  const signed=value=>Number.isFinite(Number(value))?`${Number(value)>0?'+':''}${number(value,4)}`:'—';
  const interval=Array.isArray(result.ci_95)?result.ci_95.map(signed).join(', '):'unavailable';
  return `<p class="panel-description">${escapeHtml(label)} difference (mixture − baseline): <strong>${signed(result.difference)}</strong>; paired event-bootstrap 95% interval [${interval}].</p>`;
}
async function loadResearch() {
  const container=document.createElement('div');container.id='research-content';container.className='panel data-panel';
  container.innerHTML='<h3>Predictive distribution study (V3)</h3><p class="panel-description">Loading the saved study…</p>';
  $('evidence-content').append(container);
  try {
    const report=await api('/api/v1/research');
    const aggregate=Object.entries(report.development_aggregate||{});
    const confirmation=report.confirmation||{};
    const bootstrap=confirmation.paired_event_bootstrap||{};
    const yearly=(report.development||[]).map(row=>[`${row.year} · ${row.candidate}`,row]);
    container.innerHTML=`<h3>Predictive distribution study (V3)</h3><p class="panel-description">${escapeHtml(report.selection_decision||'Compare saved predictive distributions using time-separated fitting, calibration, and evaluation periods.')}</p><p class="panel-description">Selection period: ${escapeHtml((report.selection_period||[]).join(' → '))}. Lower negative log likelihood (NLL) rewards probability assigned to observed counts. Interval widths are in strikes per athlete.</p><h4>Development comparison</h4>${aggregate.length?studyTable(aggregate,report.selected_candidate):'<p class="panel-description">No completed development results are available.</p>'}<h4>${escapeHtml(confirmation.year||2024)} confirmation comparison</h4>${studyTable(Object.entries(confirmation.metrics||{}))}${studyDifference('NLL',bootstrap.nll)}${studyDifference('MAE',bootstrap.mae)}<p class="panel-description">${number(bootstrap.events,0)} paired events; ${number(bootstrap.resamples,0)} bootstrap resamples. A negative difference favors the mixture. These results do not establish an MAE improvement. This historical year was previously inspected; it is not an untouched holdout. Prospective performance remains pending.</p>${report.warnings?.length?`<ul class="limitation-list">${report.warnings.map(w=>`<li>${escapeHtml(factText(w))}</li>`).join('')}</ul>`:''}${yearly.length?`<details class="report-details"><summary>Results by development year</summary>${studyTable(yearly)}</details>`:''}<a class="text-link" href="/api/v1/research">Download the preserved study report ↗</a>`;
  } catch(error) {
    container.innerHTML=`<h3>Predictive distribution study (V3)</h3><p class="panel-description">The saved study is unavailable: ${escapeHtml(error.message)}. The historical evaluation below remains available.</p>`;
  }
}
const evidenceObserver=new MutationObserver(()=>{
  if(!state.loaded.has('evidence'))return;
  if(!$('research-content'))loadResearch();
  if(!$('experiments-content'))loadExperiments();
});
evidenceObserver.observe($('evidence-content'),{childList:true});
function ingestFighterPhotos(photos) {
  const entries=Array.isArray(photos)?photos.map(fighter=>[fighter.fighter_id||fighter.id,fighter]):Object.entries(photos||{});
  entries.forEach(([id,fighter])=>rememberPhoto({...fighter,fighter_id:fighter.fighter_id||fighter.id||id}));
}
function rememberPhoto(fighter) {
  const id = fighter?.fighter_id || fighter?.id;
  const url = fighter?.photo_url || fighter?.image_url || fighter?.photo?.url;
  const observedAt=fighter?.photo_observed_at||fighter?.observed_at||fighter?.photo?.observed_at;
  const existing=state.photos.get(String(id));
  if(existing?.observedAt&&(!observedAt||String(observedAt)<String(existing.observedAt)))return;
  if (id && url) state.photos.set(String(id), {url, source: fighter.photo_source_url || fighter.profile_url || fighter.photo?.source_url,observedAt,provider:fighter.photo_provider||fighter.provider,alternates:fighter.alternate_photo_urls||fighter.photo?.alternate_photo_urls||[]});
}
function addExplorerPhotos(prediction) {
  const panel=$('forecast-result').querySelector('.result-panel');
  const strip=document.createElement('div');
  strip.className='explorer-athletes';
  const selected=[prediction.fighter,prediction.opponent].map(f=>({...state.fighters.find(item=>String(item.fighter_id)===String(f.id)),...f}));
  strip.innerHTML=`${selected.map((fighter,index)=>`<div>${athletePortrait(fighter,true)}<span>${escapeHtml(fighter.name)}</span><small>${index===0?'FORECAST ATHLETE':'OPPONENT'}</small></div>`).join('<span class="explorer-versus">VS</span>')}`;
  panel.prepend(strip);
  bindPortraits(strip);
  const notes=document.createElement('details');notes.className='forecast-notes';notes.innerHTML='<summary>Model & data notes</summary>';
  for(const selector of ['.warnings','.provenance']){const node=panel.querySelector(selector);if(node)notes.append(node);}
  panel.append(notes);
}
function athletePortrait(fighter, compact=false) {
  rememberPhoto(fighter);
  const identity = String(fighter.id || fighter.fighter_id || '');
  const photo = state.photos.get(identity) || {url:fighter.photo_url||fighter.image_url,source:fighter.photo_source_url};
  const url = photo?.url && safeUrl(photo.url);
  const candidates=photo?.alternates||fighter.alternate_photo_urls||[];
  const observed=Array.from(new Set([url,...(Array.isArray(candidates)?candidates:[]).map(safeUrl)].filter(Boolean)));
  const preferred=url;
  const alternates=observed.filter(candidate=>candidate!==preferred);
  const initials = (fighter.name || fighter.fighter_name || 'Athlete').split(/\s+/).filter(Boolean).slice(0,2).map(word=>word[0]).join('');
  const provider=String(photo?.provider||fighter.photo_provider||fighter.provider||'').toLowerCase();
  const headshot=preferred&&/headshots?\b/i.test(new URL(preferred).pathname);
  return `<div class="athlete-portrait ${compact?'compact':''} ${provider==='ufc'?'official-ufc-portrait':''} ${headshot?'verified-headshot':''}"><span class="portrait-initials" aria-hidden="true">${escapeHtml(initials)}</span>${preferred?`<img src="${escapeHtml(preferred)}" data-photo-fallbacks="${escapeHtml(JSON.stringify(alternates))}" alt="${escapeHtml(fighter.name||fighter.fighter_name)}" loading="lazy" decoding="async">`:''}</div>`;
}
function bindPortraits(container) {
  container.querySelectorAll('.athlete-portrait img').forEach(img=>{
    let alternates=[];
    try{alternates=JSON.parse(img.dataset.photoFallbacks||'[]').filter(value=>safeUrl(value));}catch{alternates=[];}
    const failed=()=>{
      const next=alternates.shift();
      if(next){img.closest('.athlete-portrait')?.classList?.remove('verified-headshot');img.style.objectFit='contain';img.src=next;return;}
      const caption=img.closest('.profile-portrait')?.querySelector('.profile-photo-caption');
      if(caption)caption.textContent='Photo unavailable';
      img.remove();
    };
    img.addEventListener('error',failed);
  });
}
function fixtureAthlete(fixture, side) {
  const prefix=side==='fighter'?'fighter':'opponent';
  const nested=fixture[prefix] || fixture[side==='fighter'?'fighter_a':'fighter_b'] || {};
  const athlete={id:nested.id||nested.fighter_id||fixture[`${prefix}_id`]||fixture[side==='fighter'?'fighter_a_id':'fighter_b_id'],name:nested.name||fixture[`${prefix}_name`]||fixture[side==='fighter'?'fighter_a_name':'fighter_b_name']||'Athlete TBA',photo_url:nested.photo_url||nested.image_url||fixture[`${prefix}_photo_url`],photo_source_url:nested.photo_source_url||nested.profile_url||fixture[`${prefix}_photo_source_url`]};
  rememberPhoto(athlete);
  return athlete;
}
function calendarDate(value, options={}) {
  if(!value)return 'Date TBA';
  const parsed=new Date(`${String(value).slice(0,10)}T12:00:00`);
  return Number.isNaN(parsed.getTime())?date(value):parsed.toLocaleDateString(undefined,{month:'long',day:'numeric',...options});
}
function projectionFor(fixture, athlete, index) {
  const predictions=fixture.projections||fixture.predictions||[];
  if(Array.isArray(predictions))return predictions.find(p=>String(p.fighter_id||p.fighter?.id)===String(athlete.id))||predictions[index];
  return predictions[athlete.id]||predictions[index===0?'fighter':'opponent'];
}
function fighterForecast(fixture, athlete, index) {
  const projection=projectionFor(fixture,athlete,index);
  if(!projection || fixture.projection_status==='unsupported')return '<div class="athlete-estimate unavailable">Forecast unavailable</div>';
  return `<div class="athlete-estimate"><strong>${number(projection.mean)}</strong><span>expected strikes</span><small>${number(projection.interval_low,0)}–${number(projection.interval_high,0)} <span>${number((projection.interval_level??.8)*100,0)}% range</span></small></div>`;
}
function upcomingFightCard(fixture) {
  const fighter=fixtureAthlete(fixture,'fighter'),opponent=fixtureAthlete(fixture,'opponent');
  const rounds=fixture.scheduled_rounds;
  const source=safeUrl(fixture.source_url||fixture.event_source_url);
  const confidence=fixture.time_confidence||fixture.schedule_confidence||fixture.schedule_kind;
  const exactTime=fixture.scheduled_at && !['date_only','unknown','tba'].includes(String(confidence).toLowerCase());
  const time=exactTime?new Date(fixture.scheduled_at).toLocaleTimeString(undefined,{hour:'numeric',minute:'2-digit',timeZoneName:'short'}):'Time TBA';
return `<article class="fight-card"><div class="fight-card-top"><span>${escapeHtml(fixture.division||fixture.weight_class||'UFC MATCHUP')}</span><span>${rounds?`${escapeHtml(rounds)} ROUNDS`:'ROUNDS TBA'}</span></div><div class="fight-athletes"><div class="fight-athlete">${athletePortrait(fighter,true)}<h3>${fighterProfileLink(fighter.id,fighter.name)}</h3>${fighterForecast(fixture,fighter,0)}</div><span class="card-versus">VS</span><div class="fight-athlete">${athletePortrait(opponent,true)}<h3>${fighterProfileLink(opponent.id,opponent.name)}</h3>${fighterForecast(fixture,opponent,1)}</div></div>${fixture.projection_status==='unsupported'?`<p class="card-omission">${escapeHtml(fixture.omission_reason||'This announced matchup does not yet have enough verified information for a forecast.')}</p>`:''}<div class="fight-card-footer"><span title="${escapeHtml(confidence||'Published card or segment time')}">Card start · ${escapeHtml(time)}</span>${source?`<a href="${escapeHtml(source)}" target="_blank" rel="noopener">Fight source ↗</a>`:'<span>Source unavailable</span>'}</div></article>`;
}
function renderUpcoming(report) {
  let fixtures=report.fixtures||[];
  if(!fixtures.length && Array.isArray(report.events))fixtures=report.events.flatMap(event=>(event.fixtures||event.bouts||[]).map(fixture=>({...event,...fixture,event_name:event.name||event.event_name,event_date:event.event_date||fixture.event_date})));
  fixtures=fixtures.filter(f=>!f.event_date||String(f.event_date).slice(0,10)<='2026-12-31').sort((a,b)=>String(a.event_date||a.scheduled_at).localeCompare(String(b.event_date||b.scheduled_at)));
  const grouped=new Map();
  fixtures.forEach(fixture=>{const key=fixture.event_id||`${date(fixture.event_date||fixture.scheduled_at)}:${fixture.event_name||''}`;if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(fixture);});
  const issued=report.schedule_observed_at||report.observed_at||fixtures[0]?.observed_at||report.source_as_of||report.issued_at||report.as_of||report.retrieved_at;
  const scheduleNote=report.notice||report.description||report.warning||'Only sourced, announced matchups are shown. Unannounced cards and unresolved forecasts are not filled in.';
  if(!fixtures.length){$('upcoming-content').innerHTML=`<div class="panel upcoming-empty"><p class="eyebrow">ANNOUNCEMENTS, NOT GUESSES</p><h3>No verified upcoming matchups are available.</h3><p>${escapeHtml(scheduleNote)}</p><p class="panel-description">Try Matchup Lab to explore a hypothetical pairing, or return after the announced schedule is refreshed.</p></div>`;return;}
  const events=Array.from(grouped.values()).map(group=>group.slice().reverse());
  const filters=events.map((group,index)=>`<button class="event-filter ${index===0?'active':''}" data-event-index="${index}" aria-pressed="${index===0}"><strong>${escapeHtml(calendarDate(group[0].event_date||group[0].scheduled_at,{month:'short'}))}</strong><span>${escapeHtml(group[0].event_name||'UFC event')}</span></button>`).join('');
  const sections=events.map((group,index)=>{
    const first=group[0], source=safeUrl(first.event_source_url||first.source_url);
    const when=new Date(`${date(first.event_date||first.scheduled_at)}T12:00:00`);
    return `<section class="event-group" data-calendar-event="${index}" ${index>0?'hidden':''}><div class="event-heading"><div class="event-date"><strong>${escapeHtml(when.toLocaleDateString(undefined,{month:'short'}))}</strong><span>${escapeHtml(when.getDate()||'—')}</span></div><div><p class="eyebrow">${escapeHtml(calendarDate(first.event_date||first.scheduled_at,{weekday:'long',year:'numeric'}))}</p><h3>${escapeHtml(first.event_name||first.name||'UFC event')}</h3>${first.location||first.venue?`<p>${escapeHtml(first.location||first.venue)}</p>`:''}</div>${source?`<a class="event-source" href="${escapeHtml(source)}" target="_blank" rel="noopener">Announcement ↗</a>`:''}</div><div class="fight-card-grid">${group.map(upcomingFightCard).join('')}</div></section>`;
  }).join('');
  $('upcoming-content').innerHTML=`<div class="calendar-summary"><span>${grouped.size} announced ${grouped.size===1?'event':'events'} · ${fixtures.length} matchups</span><span>Snapshot ${escapeHtml(date(issued))}</span></div><div class="event-filters" aria-label="Choose an announced event">${filters}<button class="event-filter" data-event-index="all" aria-pressed="false"><strong>All cards</strong><span>Through Dec 31</span></button></div>${sections}<div class="calendar-disclosure"><p>Research only · Not betting advice. Model estimates can be wrong. Do not use OctogonML to place bets or risk money. You are responsible for how you use these estimates. Counts and 80% ranges describe possible full-bout strike outcomes, not winner predictions.</p><details><summary>Schedule, photos & forecast details</summary><p>${escapeHtml(scheduleNote)}</p><p>Schedule snapshot ${escapeHtml(date(issued))}. Model history through ${escapeHtml(date(report.data_cutoff||report.history_cutoff))}. ${escapeHtml(report.model_version||'')}.</p><p>Fighter photos are sourced from official UFC and ESPN athlete profiles where available. Missing or unavailable photos use initials. Images identify athletes; this independent project is not affiliated with UFC or ESPN. Published times refer to a card or segment, not individual bout walkouts.</p></details></div>`;
  bindPortraits($('upcoming-content'));
  $('upcoming-content').querySelectorAll('[data-event-index]').forEach(button=>button.addEventListener('click',()=>{
    const selected=button.dataset.eventIndex;
    $('upcoming-content').querySelectorAll('[data-event-index]').forEach(item=>{item.classList.toggle('active',item===button);item.setAttribute('aria-pressed',String(item===button));});
    $('upcoming-content').querySelectorAll('[data-calendar-event]').forEach(group=>group.hidden=selected!=='all'&&group.dataset.calendarEvent!==selected);
  }));
}
async function loadUpcoming() {
  if(state.loaded.has('upcoming'))return;
  try {const report=await api('/api/v1/upcoming');state.upcoming=report;renderUpcoming(report);state.loaded.add('upcoming');}
  catch(error){$('upcoming-content').innerHTML=`<div class="panel upcoming-empty"><h3>The upcoming schedule couldn’t be loaded.</h3><p>${escapeHtml(error.message)}</p><p class="panel-description">Matchup Lab and Recent Results remain available. No placeholder fights have been added.</p><button class="secondary-button" id="retry-upcoming">Try again</button></div>`;$('retry-upcoming').addEventListener('click',loadUpcoming);}
}
const initialization=initialize();
routeLocation();
