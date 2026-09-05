#!/usr/bin/env python3
"""Controlled vehicle-only replay of the release-time/BC cost proposal.

Replays frozen confirmed inputs. The historical +2 manual adjustment is
reported separately; it is not new evidence of correctness for changed routes.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from pipeline.config import load_project_env
from pipeline.backtrack.resolver import SmartBacktrackConfig, SmartBacktrackResolver
from pipeline.backtrack.study import StudyConfig
from summarize_actor_ground_truth_metrics import VEHICLE_GT, _allowed_ids, _case_id


def wilson(hits, count):
    if not count:
        return None
    z = 1.959963984540054
    p = hits/count
    den = 1+z*z/count
    center = (p+z*z/(2*count))/den
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [center-half, center+half]


def summarize(rows, usable):
    known = {cid: _allowed_ids(VEHICLE_GT.get(cid)) for cid in usable
             if _allowed_ids(VEHICLE_GT.get(cid)) is not None}
    by_case = {cid: [r for r in rows if r['case']==cid] for cid in sorted(usable)}
    correct = {cid: any(r['vehicle_id'] in ids for r in by_case[cid])
               for cid, ids in known.items()}
    hits = sum(correct.values())
    wrong_events = sum(r['vehicle_id'] is not None and r['vehicle_id'] not in known[r['case']]
                       for r in rows if r['case'] in known)
    return dict(numeric_correct=hits, numeric_denominator=len(known),
                numeric_wilson95=wilson(hits,len(known)),
                historical_adjusted_correct=hits+len({74,174}&usable),
                historical_adjusted_denominator=len(usable),
                historical_adjusted_wilson95=wilson(hits+len({74,174}&usable),len(usable)),
                historical_adjustment_cases=sorted({74,174}&usable),
                historical_adjustment_note='Fixed prior manual +2; unknown IDs, not revalidated for new routes.',
                usable_confirmed_cases=sum(bool(v) for v in by_case.values()),
                no_confirmed_cases=[cid for cid,v in by_case.items() if not v],
                wrong_vehicle_events=wrong_events,
                null_route_events=sum(r['route_type']=='null' for r in rows),
                no_vehicle_events=sum(r['vehicle_id'] is None for r in rows),
                correctness=correct)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--clips', type=Path, default=Path('runs/grounding_truth/clip_annotations.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_project_env()
    clips = [json.loads(x) for x in args.clips.read_text().splitlines() if x.strip()]
    usable = {_case_id(r['video_filename']) for r in clips if r['video_usable']}
    inputs, hashes = [], {}
    for path in sorted(args.candidates.glob('*_backtrack_candidates.jsonl')):
        cid = _case_id(path)
        if cid not in usable:
            continue
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        for row in (json.loads(x) for x in data.splitlines() if x.strip()):
            if row.get('record_type') == 'candidate':
                task = row.get('resolver_input')
                if not task or not task.get('actor_frames'):
                    raise ValueError(f'{path} lacks frozen resolver input')
                inputs.append((cid, task))
    if not inputs:
        raise ValueError('No confirmed replay inputs')
    live = SmartBacktrackConfig.from_env(10)
    baseline = StudyConfig(name='baseline',
        dustbin_cost=live.dustbin_cost, null_vehicle_penalty=live.null_vehicle_penalty,
        direct_vehicle_penalty=live.direct_vehicle_penalty, ac_weight=live.ac_weight,
        bc_support_bonus=live.bc_support_bonus, sigma_floor_px=live.sigma_floor_px,
        two_point_prior_cost=live.two_point_prior_cost,
        max_forward_release_seconds=live.max_forward_release_seconds,
        release_window_prior_weight=live.release_window_prior_weight,
        bc_boundary_depth_weight=live.cost_config.bc_weights.get('boundary_depth',0))
    configs = [baseline,
        replace(baseline,name='normalized_bc_d04', normalized_distance_gate_vehicle=.4,
                normalize_bc_distance_time_by_gate=True),
        replace(baseline,name='release_1s',max_release_back_seconds=1),
        replace(baseline,name='combined_d04_1s',normalized_distance_gate_vehicle=.4,
                normalize_bc_distance_time_by_gate=True,max_release_back_seconds=1)]
    source_paths = [Path(__file__), Path('scripts/summarize_actor_ground_truth_metrics.py'),
                    Path('scripts/pipeline/config.py'),
                    *sorted(Path('scripts/pipeline/backtrack').glob('*.py'))]
    report = dict(schema='release-policy-replay/v1',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        input_sha256=hashes, clips_sha256=hashlib.sha256(args.clips.read_bytes()).hexdigest(),
        confirmed_events=len(inputs), configs={c.name:asdict(c) for c in configs}, trials={})
    for config in configs:
        (args.output/(config.name+'.config.json')).write_text(json.dumps(asdict(config),indent=2))
        rows=[]
        started=time.monotonic()
        resolved_configs={}
        with (args.output/(config.name+'.events.jsonl')).open('w') as out:
            for idx,(cid,task) in enumerate(inputs,1):
                fps=float(task.get('fps') or 10)
                resolved_config=config.resolver_config(fps)
                resolved_configs[str(fps)]=asdict(resolved_config)
                result=SmartBacktrackResolver(fps,resolved_config).resolve_task(dict(task))
                row=dict(case=cid,litter_id=task.get('litter_id'),
                    vehicle_id=result.vehicle_key[1] if result.vehicle_key else None,
                    vehicle_key=result.vehicle_key,route_type=result.route_type,
                    release_frame=result.release_frame,release_point=result.release_point,
                    birth_frame=task.get('birth_frame'),cost=result.total_cost,
                    actor_margins=result.actor_margins)
                rows.append(row)
                out.write(json.dumps(row,allow_nan=False)+'\n'); out.flush()
                if idx%10==0:
                    print(f'{config.name}: {idx}/{len(inputs)}',flush=True)
        summary=summarize(rows,usable)
        summary['elapsed_seconds']=time.monotonic()-started
        summary['resolved_configs_by_fps']=resolved_configs
        report['trials'][config.name]=summary
        base=report['trials']['baseline']['correctness']
        if config.name!='baseline':
            gains=[c for c,v in summary['correctness'].items() if v and not base[c]]
            losses=[c for c,v in summary['correctness'].items() if not v and base[c]]
            discordant=len(gains)+len(losses)
            p=min(1,2*sum(math.comb(discordant,k) for k in range(min(len(gains),len(losses))+1))/2**discordant) if discordant else 1
            summary.update(gains=gains,losses=losses,paired_exact_p=p)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(config.name,{k:v for k,v in summary.items() if k not in ('resolved_configs_by_fps','correctness')},flush=True)


if __name__=='__main__':
    main()
