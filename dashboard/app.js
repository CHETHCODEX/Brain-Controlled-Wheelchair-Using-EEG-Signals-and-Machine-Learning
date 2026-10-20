'use strict';
const names = {REST:'Neutral / stop',FORWARD:'Forward',TURN_LEFT:'Left',TURN_RIGHT:'Right',BACKWARD:'Backward'};
const icons = {REST:'circle-pause',FORWARD:'arrow-up',TURN_LEFT:'arrow-left',TURN_RIGHT:'arrow-right',BACKWARD:'arrow-down'};
const order = ['FORWARD','TURN_LEFT','TURN_RIGHT','BACKWARD','REST'];
const sensorNames = {front_left:'Front L',front_center:'Front',front_right:'Front R',rear_left:'Rear L',rear_center:'Rear',rear_right:'Rear R',side_left:'Left',side_right:'Right'};
const channels = ['C3','Cz','C4','FC3','FC4','CP3','CP4'];
const colors = ['#187c70','#496e9f','#a27c36','#488b84','#835a77','#678d46','#93644a'];
const byId = id => document.getElementById(id);
let state = null;
let connected = false;
const started = Date.now();
const floor = new Image();
floor.src = '/assets/hospital_floor.png';
floor.onload = drawMap;
byId('probabilities').innerHTML = order.map(intent => `<div class="probability-row" id="prob-${intent}"><span class="probability-name"><i data-lucide="${icons[intent]}"></i>${names[intent]}</span><div class="probability-bar"><div class="probability-fill"></div></div><span class="probability-number">--</span></div>`).join('');
byId('sensors').innerHTML = Object.entries(sensorNames).map(([key,label]) => `<div class="sensor" id="sensor-${key}"><span>${label}</span><strong>--</strong></div>`).join('');
if (window.lucide) window.lucide.createIcons();
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.body.dataset.view = tab.dataset.view;
  document.querySelectorAll('.tab').forEach(item => {item.classList.toggle('active', item === tab); item.setAttribute('aria-pressed', String(item === tab));});
  requestAnimationFrame(drawAll);
}));
const formatReason = value => ({command:'Validated movement',neutral:'Neutral / controlled stop',obstacle:'Front obstacle stop',rear_obstacle:'Rear obstacle stop',turn_clearance:'Turning clearance blocked',heartbeat_timeout:'Command signal lost',signal_lost_or_poor:'Poor or disconnected EEG',rest_or_low_confidence:'Low confidence / safe stop',emergency_stop:'Caregiver emergency stop',caregiver_reset:'Stopped / rearming',publisher_changed:'New input session',sensor_fault:'Sensor fault / stopped'}[value] || (value || 'Awaiting simulator').replaceAll('_',' '));
function showNotice(message) { byId('notice').textContent = message; byId('notice').hidden = !message; }
async function emergency(active) {
  if (!active && !window.confirm('Clear the caregiver emergency latch? Motion remains inhibited during the rearming interval.')) return;
  try {
    const response = await fetch('/api/emergency', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({active})});
    if (!response.ok) throw new Error('Caregiver command was not acknowledged. Pause Webots if necessary.');
    showNotice(active ? 'CAREGIVER EMERGENCY STOP LATCHED' : 'Emergency latch cleared. Rearming in neutral.');
  } catch (error) {showNotice(error.message);}
}
byId('emergency').addEventListener('click', () => emergency(true));
byId('reset').addEventListener('click', () => emergency(false));
function render(data) {
  if (!data.packet) return;
  state = data;
  const packet = data.packet;
  const robot = data.webots;
  connected = true;
  byId('source').textContent = data.simulated ? 'SIMULATED EEG' : 'EXTERNAL CLASSIFIER';
  byId('source').classList.toggle('external', !data.simulated);
  byId('wave-source').textContent = data.simulated ? 'SIMULATED' : 'EXTERNAL INPUT';
  byId('wave-source').classList.toggle('external', !data.simulated);
  byId('bridge').innerHTML = `<b></b>${robot ? 'Webots connected' : 'Awaiting Webots'}`;
  byId('bridge').classList.toggle('online', Boolean(robot));
  byId('intent').textContent = names[packet.intent] || packet.intent;
  byId('confidence').innerHTML = `${Math.round(packet.confidence*100)}<em>%</em>`;
  byId('validated').textContent = `Validated: ${names[data.validated_intent] || 'Neutral'}`;
  byId('stability').textContent = data.stable ? 'Stable consecutive predictions' : data.validated_intent === 'REST' ? 'Neutral or validation pending' : 'Validation pending';
  byId('headset').textContent = packet.headset_connected ? (data.simulated ? 'Simulated stream' : 'Connected') : 'Disconnected';
  byId('headset').classList.toggle('danger', !packet.headset_connected);
  byId('quality').textContent = `Signal quality ${Math.round(packet.signal_quality*100)}%`;
  byId('safety').textContent = data.emergency ? 'E-stop latched' : !robot ? 'Standby' : robot.intent === 'REST' ? 'Stopped' : 'Motion enabled';
  byId('safety').classList.toggle('danger', data.emergency);
  byId('safety-detail').textContent = robot ? formatReason(robot.reason) : 'Motion telemetry unavailable';
  byId('emergency').classList.toggle('latched', data.emergency);
  byId('emergency').querySelector('span').textContent = data.emergency ? 'Stop latched' : 'Emergency stop';
  byId('reset').disabled = !data.emergency;
  byId('speed').innerHTML = robot ? `${Math.abs(robot.measured_speed ?? robot.linear_velocity).toFixed(2)} <small>m/s</small>` : '-- <small>m/s</small>';
  byId('direction').textContent = robot ? names[robot.intent] || robot.intent : 'Unavailable';
  const distances = robot ? Object.values(robot.distances) : [];
  const nearest = distances.length ? Math.min(...distances) : null;
  byId('clearance').innerHTML = nearest !== null ? `${nearest.toFixed(2)} <small>m</small>` : '-- <small>m</small>';
  byId('obstacle-status').textContent = nearest === null ? 'No telemetry' : nearest <= .35 ? 'CRITICAL CLEARANCE' : nearest < .65 ? 'Obstacle nearby' : 'Clearance monitored';
  byId('obstacle-status').classList.toggle('danger', nearest !== null && nearest <= .35);
  byId('map-status').textContent = robot ? 'WEBOTS / ACTUAL POSITION TELEMETRY' : 'WEBOTS POSITION UNAVAILABLE';
  for (const key of Object.keys(sensorNames)) {
    const element = byId(`sensor-${key}`);
    const value = robot?.distances[key];
    element.querySelector('strong').textContent = value === undefined ? '--' : `${value.toFixed(2)}m`;
    element.classList.toggle('warning', value !== undefined && value <= .35);
  }
  for (const intent of order) {
    const row = byId(`prob-${intent}`);
    const value = packet.probabilities[intent] || 0;
    row.classList.toggle('active', packet.intent === intent);
    row.querySelector('.probability-fill').style.width = `${value*100}%`;
    row.querySelector('.probability-number').textContent = `${Math.round(value*100)}%`;
  }
  byId('gate').textContent = data.emergency ? 'Emergency stop has priority' : data.stable ? 'Confidence and stability checks passed' : formatReason(data.gate_reason);
  byId('gate-dot').classList.toggle('good', data.stable && !data.emergency);
  byId('classifier').textContent = data.classifier;
  byId('wave-status').textContent = data.waves ? (data.simulated ? 'Synthetic EEG / causal display filter' : 'External EEG / causal display filter') : 'No raw EEG samples supplied';
  byId('history-count').textContent = `${data.history.length} recent predictions`;
  byId('history').replaceChildren(...data.history.map(event => {
    const row = document.createElement('tr');
    const values = [new Date(event.timestamp*1000).toLocaleTimeString('en-GB'), names[event.intent] || event.intent, `${Math.round(event.confidence*100)}%`, event.source === 'simulated' ? 'SIMULATED' : 'EXTERNAL', event.accepted ? 'Validated / neutral' : 'Pending / inhibited'];
    values.forEach((value,index) => {const cell = document.createElement('td'); cell.textContent = value; if(index===4)cell.className=event.accepted?'result-good':'result-wait'; row.append(cell);});
    return row;
  }));
  if(data.emergency) showNotice('CAREGIVER EMERGENCY STOP LATCHED. Motion is inhibited.');
  else if(data.rearming) showNotice('Caregiver reset acknowledged. Neutral rearming interval.');
  else if(!packet.headset_connected) showNotice('EEG signal unavailable. Safe stop requested.');
  else if(!robot) showNotice('Webots telemetry is unavailable. Motion and obstacle status are not confirmed.');
  else showNotice('');
  drawAll();
}
function canvasContext(id) {
  const canvas = byId(id); const width = canvas.clientWidth; const height = canvas.clientHeight;
  const ratio = Math.min(window.devicePixelRatio || 1,2);
  if(canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)){canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);}
  const context = canvas.getContext('2d'); context.setTransform(ratio,0,0,ratio,0,0); context.clearRect(0,0,width,height);
  return {context,width,height};
}
function drawEEG() {
  const {context,width,height} = canvasContext('eeg'); if(!width || !height)return;
  context.fillStyle='#fcfefd';context.fillRect(0,0,width,height);
  context.strokeStyle='#edf2ef';context.lineWidth=1;
  for(let column=45;column<width;column+=30){context.beginPath();context.moveTo(column,0);context.lineTo(column,height);context.stroke();}
  const waves = connected ? state?.waves : null;
  channels.forEach((channel,index)=>{
    const center = (index+.5)*height/7;
    context.fillStyle='#718580';context.font='9px monospace';context.fillText(channel,9,center+3);
    context.strokeStyle='#e3ebe5';context.beginPath();context.moveTo(40,center);context.lineTo(width,center);context.stroke();
    if(!waves?.[index])return;
    context.strokeStyle=colors[index];context.lineWidth=1.1;context.beginPath();
    waves[index].forEach((value,sample)=>{const x=45+sample/319*(width-57);const y=center-Math.max(-40,Math.min(40,value))*(height/7/90);if(sample===0)context.moveTo(x,y);else context.lineTo(x,y);});context.stroke();
  });
  if(!waves){context.fillStyle='#60796f';context.font='12px sans-serif';context.textAlign='center';context.fillText(connected?'No raw EEG supplied':'Signal stream offline',width/2,height/2);context.textAlign='left';}
}
function drawMap() {
  const {context,width,height} = canvasContext('floorplan');if(!width || !height)return;
  if(floor.complete && floor.naturalWidth)context.drawImage(floor,0,0,width,height);
  const point=(x,y)=>[(x+9)/18*width,(6-y)/12*height];
  function block(x,y,sx,sy,color){const [left,top]=point(x-sx/2,y+sy/2);context.fillStyle=color;context.fillRect(left,top,sx/18*width,sy/12*height);}
  [-4,0,4].forEach(x=>block(x,3.9,.1,4.2,'#bdccc8'));
  [-3,3].forEach(x=>block(x,-3.9,.1,4.2,'#bdccc8'));
  [[-7,4.2,1.1,2.05],[-3,4.2,1.1,2.05]].forEach(([x,y,sx,sy])=>{block(x,y,sx,sy,'#fbfcf9');block(x,y+.7,.8,.45,'#96c4bd');});
  [[1.3,4.5,1.7,.8],[6.2,-4.5,2.7,.8],[-6.5,-4.4,1.8,.8],[-.8,-4.8,2.5,.8]].forEach(([x,y,sx,sy])=>block(x,y,sx,sy,'#7b9996'));
  block(6.2,4.4,2.1,1.2,'#6fba9a');block(-1,1.3,1.4,.5,'#517d75');block(2.7,-1.2,.55,.55,'#c48759');
  const robot=connected?state?.webots:null;
  if(robot?.position){const [x,y]=point(robot.position[0],robot.position[1]);context.save();context.translate(x,y);context.rotate(-(robot.yaw||0));context.fillStyle=state.emergency?'#c33d41':'#df9b2e';context.shadowColor='#ffffff';context.shadowBlur=7;context.fillRect(-9,-7,18,14);context.shadowBlur=0;context.fillStyle='#254640';context.fillRect(-8,-10,14,3);context.fillRect(-8,7,14,3);context.fillStyle='white';context.beginPath();context.moveTo(12,0);context.lineTo(4,-4);context.lineTo(4,4);context.fill();context.restore();}
}
function drawAll(){drawMap();drawEEG();}
new ResizeObserver(drawAll).observe(document.querySelector('.workspace'));
async function poll(){
  try {const response=await fetch('/api/state',{cache:'no-store'});if(!response.ok)throw new Error('Dashboard unavailable');const data=await response.json();if(data.service_error)throw new Error(data.service_error);render(data);}
  catch(error){connected=false;byId('bridge').innerHTML='<b></b>Dashboard offline';byId('bridge').classList.remove('online');byId('safety').textContent='Unconfirmed';byId('speed').innerHTML='-- <small>m/s</small>';showNotice('Dashboard connection lost. Wheelchair state is unconfirmed; the controller watchdog must stop motion.');drawAll();}
  const elapsed=Math.floor((Date.now()-started)/1000);byId('uptime').textContent=`${String(Math.floor(elapsed/60)).padStart(2,'0')}:${String(elapsed%60).padStart(2,'0')}`;
  window.setTimeout(poll,250);
}
poll();