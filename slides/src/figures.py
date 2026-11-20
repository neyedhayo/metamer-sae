from html import escape
import csv
import json
from pathlib import Path
from urllib.parse import quote

DATA = Path(__file__).resolve().parents[1] / 'data'
asset_references = set()


def rows(filename):
    with (DATA / filename).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def measurements(filename, label, *columns):
    return [(row[label], *(float(row[key]) for key in columns)) for row in rows(filename)]


def load_record(filename):
    return json.loads((DATA / filename).read_text(encoding='utf-8'))


COLORS = {'ink':'#17223d','source':'#17223d','standard':'#9a315e','adversarial':'#087970','slate':'#4161b6',
          'muted':'#53627e','rule':'#dce3f0','paper':'#f3f5fc','pca':'#6254ad'}
ROLE_LABELS = {'source':'Natural source','standard':'Standard','adversarial':'Adversarial'}


def image(id, role, label=None, stage=None, dynamic=False, standalone=False):
    key = '-'.join([id,role] + ([stage] if stage else []))
    asset_references.add(key)
    alt = 'Natural source image' if role == 'source' else f'{ROLE_LABELS[role]} metamer'
    dynamic_attrs = f' data-role="{role}" data-alt="{alt}"' if dynamic else ''
    alt += f' for Feather source {id}' + (f' at {stage}' if stage else '')
    img = f'<img data-photo="{key}" alt="{alt}"{dynamic_attrs}>'
    if standalone:
        return img
    text = ROLE_LABELS[role] if label is None else label
    caption = f'<figcaption><i class="dot"></i>{text}</figcaption>' if text else ''
    return f'<figure class="photo {role}">{img}{caption}</figure>'

def triptych(id, cls='', short=False, dynamic=False):
    return f'<div class="triptych {cls}">' + ''.join(
        image(id,role,ROLE_LABELS[role] if short or role=='source' else ROLE_LABELS[role]+' metamer', dynamic=dynamic)
        for role in ROLE_LABELS) + '</div>'


def metric_bar(label, value, maxvalue, cls, fmt='.3f'):
    return (f'<div class="metric-bar {cls}"><b>{label}</b><span class="track"><i style="--w:{value/maxvalue:.2%}"></i></span>'
            f'<span class="val">{value:{fmt}}</span></div>')

def svg(w,h,desc,parts):
    return f'<svg role="img" aria-label="{escape(desc, quote=True)}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg"><title>{escape(desc)}</title>{"".join(parts)}</svg>'

def t(x,y,s,size=17,fill='ink',align='start',bold=False):
    c=COLORS.get(fill,fill)
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{c}" text-anchor="{align}" font-family="Avenir Next, Arial, Helvetica" font-weight="{700 if bold else 400}">{escape(str(s))}</text>'

def line(x1,y1,x2,y2,c='rule',w=1,dash=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLORS.get(c,c)}" stroke-width="{w}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'

def rect(x,y,w,h,c,rx=0):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{COLORS.get(c,c)}"/>'

def mark(x,y,c='slate',r=5,shape='circle'):
    fill=COLORS.get(c,c)
    if shape=='diamond':
        return f'<path d="M{x},{y-r} {x+r},{y} {x},{y+r} {x-r},{y}Z" fill="{fill}"/>'
    if shape=='tick':
        return line(x,y-r,x,y+r,c,2.5)
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>'

def whisker(x0,x1,y,c='slate'):
    return line(x0,y,x1,y,c,3)+mark(x0,y,c,5,'tick')+mark(x1,y,c,5,'tick')

def media(key, alt, extra=''):
    asset_references.add(key)
    return f'<img data-photo="{key}" alt="{escape(alt, quote=True)}" {extra}>'


def diagram_data(name, data):
    return f'<script type="application/json" data-diagram="{name}">{json.dumps(data,separators=(",", ":"))}</script>'


def generation_diagram():
    record = load_record('generator.json')
    frames = record['frames']
    for frame in frames:
        media(frame['input'], '')
        for key in frame['maps']:
            media(key, '')

    names = ['Conv stem', 'Layer 1', 'Layer 2']
    shape_labels = [' × '.join(map(str, shape)) for shape in record['tensors']]
    source = image('0000','source',standalone=True)
    candidate = media(frames[0]['input'], 'Candidate pixels initialized from white noise', 'data-gen-input')
    cols = []
    for i,(layer,shape) in enumerate(zip(names,shape_labels)):
        selected = ' is-target' if i==2 else ''
        cols.append(f'''<div class="gen-column{selected}">
          <div class="gen-top"><b>{layer}</b><small>{shape}</small></div>
          <figure class="activation-stack">{media(f'generator-reference-{i}', f'Four source response maps at {layer}')}</figure>
          <figure class="activation-stack">{media(frames[0]['maps'][i], f'Four candidate response maps at {layer}', f'data-gen-map="{i}"')}</figure>
        </div>''')
    errors = [frame['error'] for frame in frames]
    points = [[11 + (frame['step']/record['steps'])**.5*193,
               8 + (1-frame['error']/max(errors))*67] for frame in frames]
    line_path = ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
    curve = svg(214,87,'Recorded normalized target error across the replay. The horizontal axis emphasizes early steps.',[
       line(9,79,207,79,'#53628e',1),f'<polyline points="{line_path}" fill="none" stroke="#9eabf1" stroke-width="2.2"/>',
       f'<circle cx="{points[0][0]}" cy="{points[0][1]}" r="4.8" fill="#c7fff4" data-gen-dot/>'])
    record['points'] = points
    return f'''<div class="gen-machine" data-generator>
      <div class="gen-banner"><span><i></i> Frozen ResNet-50</span><span>Match <code>layer2_fake_relu</code></span></div>
      <div class="gen-architecture">
        <div class="gen-column gen-inputs">
          <div class="gen-top"><b>Image pixels</b><small>Input</small></div>
          <figure>{source}<figcaption>Source · fixed</figcaption></figure>
          <figure>{candidate}<figcaption>Candidate · updated</figcaption></figure>
        </div>
        {''.join(cols)}
        <aside class="gen-loss">
          <div class="gen-top"><b>Activation error</b><small>All target-layer values</small></div>
          <span class="gen-formula">‖A − Ã‖₂ / ‖A‖₂</span>
          <output class="gen-reading" data-gen-value>{errors[0]:.3f}</output>
          {curve}
          <div class="gen-small">Recorded replay error</div>
        </aside>
      </div>
      <div class="gen-feedback"><svg viewBox="0 0 1130 31" aria-hidden="true" preserveAspectRatio="none"><path id="feedback-line" d="M1120 1 V11 Q1120 23 1104 23 H23 Q8 23 8 10 V2" stroke="#9eabf1" stroke-width="2" stroke-dasharray="8 6" fill="none"/><path d="M2 9 L8 1 14 9" fill="none" stroke="#c7fff4" stroke-width="2"/></svg><span>Backpropagate into the candidate pixels</span></div>
      <div class="gen-controls">
        <button type="button" class="replay-button" data-gen-play>▶ Replay optimization</button>
        <label class="sr-only" for="gen-scrub">Recorded iteration</label>
        <input type="range" id="gen-scrub" data-gen-scrub min="0" max="{len(frames)-1}" value="0">
        <output data-gen-step>Step 0 / {record['steps']}</output>
      </div>
      {diagram_data('generator', record)}
    </div>'''


def observer_diagram():
    rec = load_record('observer.json')
    channels = rec['channels']
    first = channels[0]
    for role in ROLE_LABELS:
        media(f'observer-{role}-pooled','')
        for index in channels:
            media(f'observer-{role}-{index}','')
    steps = [
       ('Stem','96 × 56 × 56',63,8),
       ('Stage 1','96 × 56 × 56',63,12),
       ('Stage 2','192 × 28 × 28',52,18),
       ('Stage 3','384 × 14 × 14',48,25),
       ('Stage 4','768 × 7 × 7',32,30),
    ]
    stage_grid = ''.join(f'<div class="conv-stage{" tapped" if i==3 else ""}{" downstream" if i==4 else ""}">'
            f'<span aria-hidden="true" class="conv-volume" style="--face:{face}px;--depth:{depth}px"></span>'
            f'<div><b>{name}</b><small>{shape}</small></div></div>'
            for i,(name,shape,face,depth) in enumerate(steps))
    role_buttons = ''.join(f'<button type="button" data-observer-role="{role}" aria-pressed="{str(role=="source").lower()}">{label}</button>'
                           for role,label in {**ROLE_LABELS, "source": "Source"}.items())
    channel_buttons = ''.join(f'<button type="button" data-observer-channel="{id}" aria-pressed="{str(index==0).lower()}">{id}</button>'
                              for index,id in enumerate(channels))
    return f'''<div class="observer-figure" data-observer>
      <div class="conv-sequence" aria-label="ConvNeXt-Tiny stem and stages one through four, with the Stage 3 probe highlighted">{stage_grid}</div>
      <div class="observer-probe">
        <section class="probe-input"><div class="probe-title">One image in</div>
          <div class="segmented" role="group" aria-label="Input stimulus">{role_buttons}</div>
          <img data-photo="0000-source" data-observer-input alt="Natural source image, Feather 0000">
        </section>
        <div class="probe-arrow" aria-hidden="true">→</div>
        <section class="probe-map"><div class="probe-title">Stage 3 spatial map</div>
          {media(f'observer-source-{first}', f'Measured source activation map for Stage 3 channel {first}', 'data-observer-map')}
          <div class="segmented" role="group" aria-label="Stage 3 channel"><span>Channel</span>{channel_buttons}</div>
        </section>
        <div class="probe-arrow" aria-hidden="true"><span>Average<br>each map</span>→</div>
        <section class="probe-vector"><div class="probe-title">384 channel means</div>
          {media('observer-source-pooled','Source pooled channel means arranged in a display grid','data-observer-vector')}
          <div class="activation-scale"><span>Negative</span><i></i><span>Positive</span></div>
        </section>
      </div>
    </div>'''


def topk_diagram():
    scores = [float(row['score']) for row in rows('topk-illustration.csv')]
    keep = min(6, len(scores))
    active = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    bars = ''.join(
        f'<div class="gate-column{" keeps" if index in active else ""}" data-gate-score="{score}">'
        f'<i style="--score:{score:.0%}"></i></div>' for index,score in enumerate(scores))
    return f'''<div class="topk-demo" data-topk>
      <div class="topk-flow">
        <div><span>Input</span><b>384</b><small>Centered observer values</small></div>
        <i>→</i><div><span>Encoder + ReLU</span><b>1,536</b><small>Nonnegative scores</small></div>
        <i>→</i><div class="gate-step"><span>TopK gate</span><b>64</b><small>Values kept in the pilot</small></div>
        <i>→</i><div><span>Decoder</span><b>384</b><small>Reconstructed values</small></div>
      </div>
      <figure class="gate-lab">
        <div class="gate-head"><figcaption><b>Try the TopK gate</b><span>{len(scores)} illustrative latent scores</span></figcaption>
          <label for="gate-count">Keep <output data-gate-value>{keep}</output></label>
          <input id="gate-count" type="range" min="1" max="{len(scores)}" value="{keep}" data-gate-count aria-label="Number of illustrative latent scores to keep">
        </div>
        <div class="gate-bars" style="--gate-size:{len(scores)}" aria-label="Purple bars show the largest retained scores, pale bars are set to zero">{bars}</div>
        <div class="gate-legend"><span><i></i> Retained</span><span><i></i> Set to zero</span>
          <span data-gate-summary>{keep} of {len(scores)} retained</span></div>
      </figure>
    </div>'''


def reconstruction_comparison():
    data = load_record('reconstructions.json')
    cases = data['cases']
    for case in cases:
        for role in ROLE_LABELS:
            image(case,role,standalone=True)
            for model in ('raw','pca16','pca64','sae'):
                media(f'reconstruction-{case}-{role}-{model}','')
                if model!='raw':
                    media(f'reconstruction-{case}-{role}-{model}-residual','')
    first = next(iter(cases))
    selectors = ''.join(f'<button type="button" data-recon-case="{case}" aria-pressed="{str(case==first).lower()}">{case}</button>'
                        for case in cases)
    headings = '''<div class="recon-label">Input image</div>
        <div class="recon-label">Original <span>384 observer channels</span></div>
        <div class="recon-label recon-pca"><strong data-recon-pca-heading>PCA reconstruction</strong> <span data-recon-pca-label>16 components</span></div>
        <div class="recon-label"><strong data-recon-sae-heading>TopK SAE reconstruction</strong> <span>64 active latents</span></div>'''
    rows = []
    for role,label in ROLE_LABELS.items():
        source = image(first, role, standalone=True)
        source = source.replace('<img ',f'<img data-recon-input="{role}" ',1)
        heading = f'<div class="recon-input {role}">{source}<b>{label}</b></div>'
        maps = []
        for model,title in (('raw','Original'),('pca16','PCA reconstruction'),('sae','TopK SAE reconstruction')):
            prefix = f'reconstruction-{first}-{role}-{model}'
            attrs = f'data-recon-map="{model}" data-recon-role="{role}"'
            plot = media(prefix,f'{title} for {label.lower()} in triplet {first}',attrs)
            if model=='raw':
                stat='<span class="recon-stat">Centered activation</span>'
            else:
                stat=f'<span class="recon-stat" data-recon-stat="{role}-{model}">MSE {cases[first][role][model]:.4f}</span>'
            maps.append(f'<figure class="recon-panel{" recon-panel-pca" if model=="pca16" else ""}">{plot}{stat}</figure>')
        rows.append(heading+''.join(maps))

    activation = data['activation_limit']
    residual = data['residual_limit']
    return f'''<figure class="recon-lab" data-reconstruction-lab>
      <div class="recon-toolbar">
        <div class="segmented" role="group" aria-label="Feather triplet"><span>Feather triplet</span>{selectors}</div>
        <div class="segmented" role="group" aria-label="PCA component count"><span>PCA</span>
          <button type="button" data-recon-pca="pca16" aria-pressed="true">16 PCs</button>
          <button type="button" data-recon-pca="pca64" aria-pressed="false">64 PCs</button>
        </div>
        <div class="segmented" role="group" aria-label="Heatmap view">
          <button type="button" data-recon-view="activation" aria-pressed="true">Activations</button>
          <button type="button" data-recon-view="residual" aria-pressed="false">Absolute error</button>
        </div>
      </div>
      <div class="recon-grid">{headings}{''.join(rows)}</div>
      <figcaption class="recon-legend">
        <div><b>Centered activation</b><span>−{activation:.2f}</span><i class="signed-scale"></i><span>+{activation:.2f}</span></div>
        <div data-recon-error-scale hidden><b>Absolute error</b><span>0</span><i class="residual-scale"></i><span>{residual:.2f}</span></div>
      </figcaption>
      {diagram_data('reconstruction',data)}
    </figure>'''


def pca_variance_chart():
    pcs = [(int(row['components']), float(row['cumulative_train_variance_fraction'])) for row in rows('pca-variance.csv')]
    xf = lambda v: 70 + v/64*500
    yf = lambda v: 260 - (v-.92)/.065*200
    parts = [t(70,34,'Cumulative variance explained',20,bold=True)]
    for tick in [.92,.94,.96,.98]:
        y=yf(tick);parts += [line(70,y,570,y),t(60,y+5,f'{tick:.0%}',14,'muted','end')]
    parts += [line(70,260,570,260),'<'+'polyline points="'+' '.join(f'{xf(x):.2f},{yf(y):.2f}' for x,y in pcs)
              +'" fill="none" stroke="#6254ad" stroke-width="3"/>']
    for pc,value in pcs:
        parts.append(mark(xf(pc),yf(value),'pca',7 if pc in (16,64) else 3))
    for pc in [0,16,32,48,64]:
        parts.append(t(xf(pc),281,pc,15,'muted','middle'))
    parts.append(t(571,309,'Components retained',16,'muted','end'))
    return svg(620,317,'Cumulative train variance by retained PCA components.',parts)


def training_chart():
    sweep = measurements('sae-validation-sweep.csv', 'configuration', 'mean_validation_mse')
    parts=[]
    for i,(label,mse) in enumerate(sweep):
        y=18+i*66;c='slate' if i==3 else 'pca'
        parts += [t(0,y+23,label,20,c,bold=i==3),rect(131,y,420,31,'paper',3),rect(131,y,420*mse/.0026,31,c,3),
                  t(667,y+22,f'{mse:.6f}',18,c,'end',True)]
    checkpoints = ''.join(
        '<div class="checkpoint">'
        + escape(row['label'])
        + f'<strong>{float(row["validation_mse"]):.6f}</strong><span class="sub">'
        + f'Seed {int(row["seed"])} · best epoch {int(row["best_epoch"])} · '
        + f'{int(row["active_validation_features"]):,} active on validation</span></div>'
        for row in rows('sae-selected-checkpoints.csv')
    )
    return ('<div class="training"><figure class="chart"><figcaption class="label">Natural-image validation MSE ↓</figcaption>'
            +svg(683,260,'Mean natural-image validation mean squared error by SAE setting.',parts)
            +'</figure><div class="checkpoints">'+checkpoints+'</div></div>')


def feature_atlas():
    samples = sorted(measurements('latent-1157-source-activations.csv', 'triplet_id', 'activation'), key=lambda row: row[1], reverse=True)
    sample_buttons = ''.join(
        f'<button class="sample" type="button" data-sample="{id}" aria-pressed="{str(i==0).lower()}" aria-label="Show triplet {id}, source activation {value:.3f}">'
        +image(id,'source',standalone=True)
        +f'<span class="value"><span>{id}</span><b>{value:.3f}</b></span><div class="activation"><i style="--v:{value/1.4:.2%}"></i></div></button>'
        for i,(id,value) in enumerate(samples))

    means = {row['condition']: float(row['mean_activation']) for row in rows('latent-1157-condition-means.csv')}
    feat_stats = ''.join(metric_bar(label,means[role],.8,role)
                         for role,label in (('source','Source'),('standard','Standard'),('adversarial','Adversarial')))
    first = samples[0][0]
    for id,_ in samples:
        for role in ROLE_LABELS:
            image(id,role,standalone=True)
    return ('<div class="feature-layout" data-atlas><div><h3>Source images ranked by activation</h3><div class="atlas">'+sample_buttons
            +f'</div></div><div class="feature-detail"><h3>Selected triplet · <span data-triplet-id>Feather {escape(first)}</span></h3>'
            +triptych(first,short=True,dynamic=True)
            +f'<div class="feature-stats"><h3>Mean activation across all {len(samples)} triplets</h3>'+feat_stats+'</div></div></div>')


def distance_comparison():
    effects = paired_results('L2')
    parts=[]
    # Each range is only a within-row comparison. Labels expose exact units by representation.
    def dumb_x(v): return 147 + v/4*371
    for i,(label,standard,adversarial,delta,low,high) in enumerate(effects):
        y=35+i*47
        sa,ad=dumb_x(standard),dumb_x(adversarial)
        parts += [t(0,y+6,label,20,bold=True), line(147,y,518,y),
                  line(ad,y,sa,y,'#aeb7d4',3), mark(ad,y,'adversarial',7),mark(sa,y,'standard',8,'diamond'),
                  t(ad,y-13,f'{adversarial:.2f}',16,'adversarial','middle',True),
                  t(sa,y-13,f'{standard:.2f}',16,'standard','middle',True)]
    means_plot = svg(570,251,'Mean source-to-standard and source-to-adversarial distances within each representation.',parts)

    def diff_x(v): return 20 + v/2.25*390
    parts=[]
    for tick in [0,0.5,1,1.5,2]:
        x=diff_x(tick);parts += [line(x,16,x,235,'#919ec9' if tick==0 else 'rule',1,'4 5' if tick==0 else ''),t(x,254,f'{tick:g}',14,'muted','middle')]
    for i,(label,standard,adversarial,delta,low,high) in enumerate(effects):
        y=35+i*47
        parts += [whisker(diff_x(low),diff_x(high),y),mark(diff_x(delta),y,'slate',6),
                  t(495,y+5,f'+{delta:.2f}',18,'slate','end',True)]
    forest=svg(510,266,'Paired mean differences and 95% bootstrap intervals, standard minus adversarial. All five intervals are above zero.',parts)
    return results_diagram(means_plot, forest)


def results_diagram(means_plot, forest):
    raw = load_record('observer.json')['0116']['l2']
    ruler_max = 1.6
    rails = ''.join(
        f'<div class="case-rail {role}"><span>{ROLE_LABELS[role]}</span>'
        f'<span class="case-track" style="--length:{value/ruler_max:.2%}">{image("0116",role,standalone=True)}</span>'
        f'<b>{value:.2f}</b></div>'
        for role,value in zip(('standard','adversarial'),raw)
    )
    return f'''<div class="result-explainer">
     <div class="result-example">
       <header><h3>One triplet</h3><small>Feather 0116 · raw 384-D</small></header>
       <div class="case-source">{image('0116','source',standalone=True)}<span>Start at the source activation</span></div>
       {rails}
       <div class="case-axis"><span>0</span><span>1.6 L2</span></div>
       <div class="case-explain">Each rail measures the length of the change in the observer’s 384 values.</div>
     </div>
     <div class="result-summary" data-switcher>
       <header><h3>All 32 primary triplets</h3><div class="result-count">32/32 <small>each space</small></div></header>
       <div class="result-chart-head"><div class="segmented" role="group" aria-label="Result chart">
         <button type="button" data-panel-button="means" aria-pressed="true">Mean distances</button>
         <button type="button" data-panel-button="ci" aria-pressed="false">Paired differences + 95% CI</button>
       </div></div>
       <div class="result-panel chart" data-panel="means">{means_plot}{legend()}</div>
       <div class="result-panel chart" data-panel="ci" hidden>{forest}<p>Standard minus adversarial · Paired bootstrap</p></div>
     </div>
    </div>'''


def support_chart():
    support = paired_results('Support Jaccard')
    parts=[]
    def support_x(v): return 103 + v*540
    for tick in [0,.25,.5,.75,1]:
        x=support_x(tick);parts += [line(x,29,x,200),t(x,223,f'{tick:g}',15,'muted','middle')]
    for i,(label,standard,adversarial,delta,low,high) in enumerate(support):
        y=80+94*i
        x0,x1=support_x(standard),support_x(adversarial)
        parts += [t(0,y+5,label,18,bold=True),line(x0,y,x1,y,'#aeb7d4',3),
                  mark(x0,y,'standard',8,'diamond'),mark(x1,y,'adversarial',8),
                  t(x0-10,y-15,f'{standard:.3f}',16,'standard','end',True),
                  t(x1+10,y-15,f'{adversarial:.3f}',16,'adversarial','start',True),
                  t(655,y+6,f'+{delta:.3f}  [{low:.3f}, {high:.3f}]',16,'slate','start',True)]
    parts += [t(105,250,'Mean source–metamer Jaccard',16,'muted'),t(655,25,'Paired Δ [95% CI]',16,'slate',bold=True)]
    return svg(855,260,'Active-support Jaccard by condition and SAE seed, with paired differences and 95% intervals.',parts)


def cosine_chart():
    cosine = measurements('pilot-cosine-gain.csv', 'representation', 'mean', 'median', 'q10', 'q90')
    def cos_x(v): return 100 + v/.44*405
    parts = []
    for tick in [0,.1,.2,.3,.4]:
        x=cos_x(tick);parts += [line(x,10,x,237),t(x,258,f'{tick:g}',14,'muted','middle')]
    for i,(label,mean,median,low,high) in enumerate(cosine):
        y=27+i*46
        parts += [t(0,y+5,label,17,bold=True),whisker(cos_x(low),cos_x(high),y,'pca'),
                  mark(cos_x(mean),y,'slate',6,'diamond'),mark(cos_x(median),y,'adversarial',7,'tick')]
    parts += [t(101,285,'Adversarial − standard',15,'muted'),t(100,310,'◆ Mean     | Median     — 10th–90th percentiles',14,'muted')]
    return svg(522,318,'Cosine gain mean, median and 10th to 90th percentiles by representation.',parts)


def reconstruction_chart():
    recon = [
        (row['representation'],row['condition'],float(row['mean_delta_mse_x1000']),
         float(row['ci_low_x1000']),float(row['ci_high_x1000']))
        for row in rows('pilot-reconstruction-delta.csv')
    ]
    def rec_x(v): return 365+(v+.60)/1.0*623
    parts=[]
    for tick in [-.6,-.4,-.2,0,.2,.4]:
        x=rec_x(tick);parts += [line(x,15,x,348,'#919ec9' if tick==0 else 'rule',1.5 if tick==0 else 1,'5 5' if tick==0 else ''),
                               t(x,369,f'{tick:g}',15,'muted','middle')]
    for i,(name,role,value,low,high) in enumerate(recon):
        y=36+i*43
        parts += [t(19,y+6,name if i%2==0 else '',20,bold=True),
                  t(149,y+6,ROLE_LABELS[role],17,role),
                  whisker(rec_x(low),rec_x(high),y,role),mark(rec_x(value),y,role,6,'diamond' if role=='standard' else 'circle'),
                  t(1110,y+6,f'{value:+.3f}',18,role,'end',True)]
    parts += [t(367,403,'Mean metamer − source reconstruction MSE (×10⁻³)',18,'muted')]
    return svg(1140,410,'Paired mean metamer-minus-source reconstruction error with 95% intervals.',parts)


def legend():
    return '<div class="legend"><span class="standard"><i></i>Standard</span><span class="adversarial"><i></i>Adversarial</span></div>'


def pca_variance(components):
    return next(f'{float(row["cumulative_train_variance_fraction"]):.2%}'
                for row in rows('pca-variance.csv') if int(row['components']) == components)


def paired_results(metric):
    return [
        (row['representation'], *(float(row[col]) for col in (
            'standard_mean', 'adversarial_mean', 'paired_difference', 'ci_low', 'ci_high')))
        for row in rows('pilot-paired-results.csv') if row['metric'] == metric
    ]


def chart_data_download():
    content = quote((DATA / 'pilot-paired-results.csv').read_text(encoding='utf-8'))
    return ('<a class="data-download" download="metamer-sae-pilot-summary.csv" '
            f'href="data:text/csv;charset=utf-8,{content}">Download chart data ↓</a>')


def spatial_distances():
    return ''.join(metric_bar(ROLE_LABELS[role], value, .26, role) for role, value in
                   measurements('pilot-spatial-relative-l2.csv', 'condition', 'mean_relative_l2'))


RENDERERS = {
    'GENERATION_DEMO': generation_diagram,
    'OBSERVER_DEMO': observer_diagram,
    'TOPK_DEMO': topk_diagram,
    'PCA_VARIANCE_CHART': pca_variance_chart,
    'PCA_16_VARIANCE': lambda: pca_variance(16),
    'PCA_64_VARIANCE': lambda: pca_variance(64),
    'RECONSTRUCTION_DEMO': reconstruction_comparison,
    'TRAINING_CHART': training_chart,
    'FEATURE_ATLAS': feature_atlas,
    'DISTANCE_COMPARISON': distance_comparison,
    'CHART_DATA_DOWNLOAD': chart_data_download,
    'SUPPORT_CHART': support_chart,
    'FIRST_SUPPORT_GAIN': lambda: f'{paired_results("Support Jaccard")[0][3]:+.3f}',
    'COSINE_CHART': cosine_chart,
    'SPATIAL_DISTANCE_BARS': spatial_distances,
    'RECONSTRUCTION_CHART': reconstruction_chart,
}
