"""Classify local optical tables by documented use; leave sources untouched."""
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shkuratov_model as model

ROOT = Path(__file__).resolve().parents[1]
OPCON = Path('C:/Users/2017j/Downloads/opcon')


def numeric_rows(path):
    if path.suffix.lower() in {'.pro', '.tgz', '.gz', '.xml', '.md', '.json'}:
        return False
    for line in path.read_text(encoding='utf-8-sig', errors='replace').splitlines():
        parts = re.split(r'[\s,]+', line.strip())
        try:
            if len(parts) >= 2:
                for value in parts:
                    float(value.replace('D', 'E').replace('d', 'e'))
                return True
        except ValueError:
            continue
    return path.suffix.lower() in {'.lnk', '.yml'}


def main():
    uses = {}
    original = model.numeric_table

    def trace(code, purpose):
        paths = []

        def tracked(path, columns):
            paths.append(Path(path).resolve())
            return original(path, columns)

        model.numeric_table = tracked
        try:
            model.load_opcon(OPCON, code)
        finally:
            model.numeric_table = original
        for path in paths:
            if path.exists():
                uses.setdefault(path, set()).add(purpose)

    scan = json.loads((ROOT / 'validation/shkuratov_material_scan.json').read_text())
    for item in scan['results']:
        if item['status'] == 'passed':
            trace(item['code'], '数値テストのみ：129材料の有限性・値域検査')
    for code in ('Gser', 'opx', 'cpx', 'olv'):
        trace(code, '前進計算例・積分収束／界面モデル比較')
    trace('Sz', 'Clark HS318表によるHS8・HS318実測比較（固定設定）')
    for code in ('K', 'M', 'Gser', 'Gcro'):
        trace(code, '読込・JWST波長被覆の確認（フィット未実施）')
    trace('T', '読込・別配布Titan tholin表との照合')
    uses.setdefault((OPCON / 'serp_chlor_mooney85').resolve(), set()).add(
        '読込・原論文の表／JWST波長被覆の確認')

    main_files = {
        'roush2021_calcite.csv': 'calcite前進計算例・実測／モデル／パラメータ比較',
        'roush2021_dolomite.csv': 'dolomite実測／モデル／パラメータ比較',
        'magnetite_Querry1985.yml': 'magnetite実測／モデル／パラメータ比較',
        'mg_serpentine_proxy_composite.lnk': '蛇紋石実測／モデル／パラメータ比較（従来proxy）',
    }
    check_files = {
        'roush2021_magnesite.csv', 'organics_tholin_khare1984.lnk',
        'dolomite_querry1987_o.yml', 'dolomite_querry1987_e.yml',
    }
    full = ROOT / 'data/optical_constants_full'
    for name, purpose in main_files.items():
        uses.setdefault((full / name).resolve(), set()).add(purpose)
    for name in check_files:
        uses.setdefault((full / name).resolve(), set()).add('読込・数値／出典／JWST波長被覆の確認のみ')
    # Copies are listed separately so an archived candidate is not mistaken for
    # an active input of the measured-spectrum comparison.
    copy_paths = {
        ROOT / 'data/serpentine_optical_constants/Gser_nk_original.txt': 'Gser_nkの保存コピー',
        ROOT / 'data/serpentine_optical_constants/serpentine_glotch2007_nk.csv': 'Gser_nkの変換コピー',
    }
    for path, purpose in copy_paths.items():
        uses.setdefault(path.resolve(), set()).add('保存・変換のみ：' + purpose)

    candidates = ROOT / 'data/serpentine_candidates'
    records = []
    files = list(OPCON.rglob('*'))
    for directory in (full, candidates, ROOT / 'data/serpentine_optical_constants'):
        files.extend(directory.rglob('*'))
    for path in sorted(set(p.resolve() for p in files if p.is_file())):
        if not numeric_rows(path):
            continue
        purposes = sorted(uses.get(path, set()))
        if path.parent == candidates and path.name != 'Materials_List_original.txt':
            purposes = ['候補として保存・変換検査のみ。計算への採用前']
            status = '未使用'
        else:
            status = '使用中' if purposes else '未使用'
        if any('前進計算' in p or 'パラメータ比較' in p or '実測比較' in p for p in purposes):
            level = '計算・比較'
        elif status == '使用中':
            level = 'テスト・読込確認・保存のみ'
        else:
            level = '計算未使用'
        location = 'Downloads/opcon' if path.is_relative_to(OPCON) else 'プロジェクト内'
        label = str(path.relative_to(OPCON if location == 'Downloads/opcon' else ROOT)).replace('\\', '/')
        records.append({'location': location, 'file': label, 'path': str(path),
                        'status': status, 'use_level': level,
                        'purpose': '；'.join(purposes) or '現在の計算・検証での参照を確認できない'})
    output = ROOT / 'data/光学定数_使用状況'
    output.mkdir(exist_ok=True)
    for status, name in (('使用中', '使用中の光学定数.md'), ('未使用', '未使用の光学定数.md')):
        selected = [r for r in records if r['status'] == status]
        lines = [f'# {status}のローカル光学定数', '', '調査日：2026-09-27。元ファイルは移動・変更していない。',
                 '「使用中」は計算例・比較に加え、実行結果が保存された数値テスト・読込検査を含む。用途欄で区別する。',
                 '「未使用」は現在の処理に未採用という意味で、利用不適切という判断ではない。',
                 'nだけ／kだけ／吸収係数・誘電関数などの光学関連表も含む。単位と列の意味は個別確認が必要。', '',
                 '| 保存場所 | ファイル | 使用区分 | 用途 |', '| --- | --- | --- | --- |']
        for row in selected:
            link = row['path'].replace('\\', '/')
            lines.append(f"| {row['location']} | [{row['file']}](<{link}>) | {row['use_level']} | {row['purpose']} |")
        (output / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / '全件一覧.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({status: sum(r['status'] == status for r in records)
                      for status in ('使用中', '未使用')}, ensure_ascii=False))
    print('計算・比較', sum(r['use_level'] == '計算・比較' for r in records))


if __name__ == '__main__':
    main()
