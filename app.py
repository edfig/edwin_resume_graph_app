import json, datetime, io, math, itertools
import streamlit as st
import networkx as nx
import pandas as pd
from pyvis.network import Network
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Edwin Figueroa — Career Knowledge Graph", layout="wide")

DATA_PATH = Path(__file__).parent / "resume_graph_data.json"

@st.cache_data
def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)

data = load_data()

GROUP_COLORS = {
    "Person": "#ff69b4",
    "Company": "#1f77b4",
    "Role": "#2ca02c",
    "Skill": "#ff7f0e",
    "Achievement": "#9467bd",
    "Education": "#d62728",
    "Year": "#cccccc",
}

# ========================= Helpers =========================
def parse_date(s):
    if not s:
        return None
    return datetime.datetime.fromisoformat(s)

def role_year_span(role):
    s = parse_date(role.get("start"))
    e = parse_date(role.get("end")) or datetime.datetime.today()
    return s.year if s else None, e.year if e else None

def score_significance(node_attrs):
    # Score node significance for default sizing.
    # Person > Role > Company > Education > Skill > Achievement.
    # Recent roles get a small bump.
    group = node_attrs.get("group")
    base = {"Person": 5.0, "Role": 3.5, "Company": 3.0, "Education": 2.5, "Skill": 2.0, "Achievement": 1.5}.get(group, 1.0)
    if group == "Role":
        title = node_attrs.get("label", "")
        years = [int(x) for x in "".join([c if c.isdigit() else " " for c in node_attrs.get("title","")]).split() if x.isdigit()]
        if years:
            yr = max(years)
            delta = max(0, datetime.datetime.today().year - yr)
            base += max(0, 3 - min(delta,3)) * 0.3  # up to +0.9 for most recent
    return base

def build_graph(filters, show_year_nodes=False):
    G = nx.DiGraph()
    # Add person if Person type is selected
    person = data["person"]
    if "Person" in filters.get("node_types", []):
        G.add_node(person["id"], label=person["label"], group="Person", title=f"{person['title']} — {person['location']}", color=GROUP_COLORS.get("Person"))

    # Companies
    company_lookup = {c["id"]: c for c in data["companies"]}
    for c in data["companies"]:
        if filters["companies"] and c["name"] not in filters["companies"]:
            continue
        if "Company" in filters.get("node_types", []):
            G.add_node(c["id"], label=c["name"], group="Company", title=c.get("location",""), color=GROUP_COLORS.get("Company"))
            if "Person" in filters.get("node_types", []):
                G.add_edge(person["id"], c["id"], label="worked_at", color="#999")

    # Roles
    for r in data["roles"]:
        comp = company_lookup[r["company_id"]]
        if filters["companies"] and comp["name"] not in filters["companies"]:
            continue
        rs, re = role_year_span(r)
        if filters["start_year"] and rs and rs < filters["start_year"]:
            if not (re and re >= filters["start_year"]):
                continue
        if filters["end_year"] and re and re > filters["end_year"]:
            if not (rs and rs <= filters["end_year"]):
                continue
        if filters["skills"] and not any(s in r["skills"] for s in filters["skills"]):
            continue

        role_node_id = r["id"]
        if "Role" in filters.get("node_types", []):
            G.add_node(role_node_id,
                      label=r["title"],
                      group="Role",
                      title=f"{r['title']} @ {comp['name']} ({rs}–{re or 'Present'})",
                      color=GROUP_COLORS.get("Role"))
            if "Company" in filters.get("node_types", []):
                G.add_edge(r["company_id"], role_node_id, label="has_role", color="#bbb")

        for i, h in enumerate(r["highlights"]):
            if "Achievement" in filters.get("node_types", []):
                a_id = f"ach:{role_node_id}:{i}"
                G.add_node(a_id, label="• " + h[:40] + ("…" if len(h)>40 else ""), group="Achievement", title=h, color=GROUP_COLORS.get("Achievement"))
                if "Role" in filters.get("node_types", []):
                    G.add_edge(role_node_id, a_id, label="achieved", color="#c4d")
        for s in r["skills"]:
            if filters["skills"] and s not in filters["skills"]:
                continue
            if "Skill" in filters.get("node_types", []):
                sid = "skill:" + s.lower().replace(" ","_").replace("/","_")
                G.add_node(sid, label=s, group="Skill", title=s, color=GROUP_COLORS.get("Skill"))
                if "Role" in filters.get("node_types", []):
                    G.add_edge(role_node_id, sid, label="used", color="#7bc")

    for e in data["education"]:
        if "Education" in filters.get("node_types", []):
            eid = e["id"]
            G.add_node(eid, label=e["school"], group="Education", title=e["degree"], color=GROUP_COLORS.get("Education"))
            if "Person" in filters.get("node_types", []):
                G.add_edge(data["person"]["id"], eid, label="studied_at", color="#9a9")

    if show_year_nodes:
        all_years = set()
        for r in data["roles"]:
            rs, re = role_year_span(r)
            if rs: all_years.add(rs)
            if re: all_years.add(re)
        
        sorted_years = sorted(list(all_years))
        for i, year in enumerate(sorted_years):
            if "Year" in filters.get("node_types", []):
                G.add_node(f"year_{year}", label=str(year), group="Year", color=GROUP_COLORS.get("Year"), size=10)
                if i > 0:
                    G.add_edge(f"year_{sorted_years[i-1]}", f"year_{year}", label="", color="#ccc")

        for r in data["roles"]:
            rs, _ = role_year_span(r)
            if rs and f"year_{rs}" in G:
                 G.add_edge(f"year_{rs}", r["id"], label="started_in", color="#ccc")
    else:
        sorted_roles = sorted(data["roles"], key=lambda r: parse_date(r["start"]) or datetime.datetime(1900,1,1))
        for i in range(len(sorted_roles)-1):
            G.add_edge(sorted_roles[i]["id"], sorted_roles[i+1]["id"], label="next", color="#ddd")
    return G

def chronological_layout(G, center=(0,0)):
    # Deterministic left-to-right timeline for Role nodes; attach related nodes nearby.
    pos = {}
    
    # Calculate dynamic spacing based on label lengths and number of nodes
    def calculate_gaps(nodes):
        max_label_len = max((len(G.nodes[n].get("label", "")) for n in nodes), default=10)
        num_nodes = len(nodes)
        # Base the x_gap on the longest label to prevent overlap
        x_gap = max(300, max_label_len * 15)  # minimum 300px, or 15px per character
        # Increase y_gap for more vertical separation when there are many nodes
        y_gap = max(160, min(200, 160 + num_nodes * 5))  # increases with node count but caps at 200
        return x_gap, y_gap
    
    # Default gaps if no nodes
    x_gap, y_gap = 300, 160
    year_nodes = [n for n, a in G.nodes(data=True) if a.get("group") == "Year"]
    if year_nodes:
        sorted_years = sorted(year_nodes, key=lambda n: int(G.nodes[n]['label']))
        x_gap, y_gap = calculate_gaps(sorted_years)  # Calculate gaps based on year nodes
        
        x = center[0] - (len(sorted_years)-1)*x_gap/2
        y = center[1]
        for i, yn in enumerate(sorted_years):
            pos[yn] = (x + i*x_gap, y)
        
        for yn in sorted_years:
            connected_roles = [c for c in G.successors(yn) if G.nodes[c].get("group") == "Role"]
            if connected_roles:
                role_x_gap, role_y_gap = calculate_gaps(connected_roles)  # Calculate gaps for roles
                for i, rn in enumerate(connected_roles):
                    pos[rn] = (pos[yn][0] + (i - (len(connected_roles)-1)/2) * role_x_gap/2, pos[yn][1] + y_gap)
                    
                    # Position related nodes with their own spacing
                    ach = [c for c in G.successors(rn) if G.nodes[c].get("group") == "Achievement"]
                    skl = [c for c in G.successors(rn) if G.nodes[c].get("group") == "Skill"]
                    
                    if ach:
                        # Calculate achievement spacing - minimum 200px between nodes
                        ach_x_gap = max(200, max(len(G.nodes[a].get("label", "")) * 20 for a in ach))
                        total_ach_width = ach_x_gap * (len(ach) - 1)
                        ach_start_x = pos[rn][0] - total_ach_width/2
                        for j, a in enumerate(ach):
                            pos[a] = (ach_start_x + j*ach_x_gap, pos[rn][1] + role_y_gap*1.5)
                            
                    if skl:
                        # Calculate skill spacing - minimum 200px between nodes
                        skl_x_gap = max(200, max(len(G.nodes[s].get("label", "")) * 20 for s in skl))
                        total_skl_width = skl_x_gap * (len(skl) - 1)
                        skl_start_x = pos[rn][0] - total_skl_width/2
                        for j, s in enumerate(skl):
                            pos[s] = (skl_start_x + j*skl_x_gap, pos[rn][1] - role_y_gap*1.5)

    else:
        role_nodes = [n for n, a in G.nodes(data=True) if a.get("group") == "Role"]
        if role_nodes:
            def year_from_title(nid):
                t = G.nodes[nid].get("title","")
                nums = [int(x) for x in "".join([c if c.isdigit() else " " for c in t]).split() if x.isdigit()]
                return nums[0] if nums else 1900
                
            role_nodes = sorted(role_nodes, key=lambda n: year_from_title(n))
            x_gap, y_gap = calculate_gaps(role_nodes)  # Calculate gaps based on role nodes
            
            x = center[0] - (len(role_nodes)-1)*x_gap/2
            y = center[1]
            
            for i, rn in enumerate(role_nodes):
                pos[rn] = (x + i*x_gap, y)
                ach = [c for c in G.successors(rn) if G.nodes[c].get("group") == "Achievement"]
                skl = [c for c in G.successors(rn) if G.nodes[c].get("group") == "Skill"]
                
                if ach:
                    # Calculate achievement spacing - minimum 200px between nodes
                    ach_x_gap = max(200, max(len(G.nodes[a].get("label", "")) * 20 for a in ach))
                    total_ach_width = ach_x_gap * (len(ach) - 1)
                    ach_start_x = x + i*x_gap - total_ach_width/2
                    for j, a in enumerate(ach):
                        pos[a] = (ach_start_x + j*ach_x_gap, y + y_gap*1.5)
                        
                if skl:
                    # Calculate skill spacing - minimum 200px between nodes
                    skl_x_gap = max(200, max(len(G.nodes[s].get("label", "")) * 20 for s in skl))
                    total_skl_width = skl_x_gap * (len(skl) - 1)
                    skl_start_x = x + i*x_gap - total_skl_width/2
                    for j, s in enumerate(skl):
                        pos[s] = (skl_start_x + j*skl_x_gap, y - y_gap*1.5)

    # Calculate final gaps for remaining nodes
    all_positioned_nodes = list(pos.keys())
    if all_positioned_nodes:
        x_gap, y_gap = calculate_gaps(all_positioned_nodes)
    
    for n, a in G.nodes(data=True):
        if n not in pos:  # Only position nodes that haven't been positioned yet
            if a.get("group") == "Person":
                pos[n] = (center[0] - 2*x_gap, y - 2*y_gap)
            elif a.get("group") == "Company":
                # Find connected role to position company node
                connected_role = next((s for s in G.successors(n) if G.nodes[s].get("group") == "Role" and s in pos), None)
                if connected_role:
                    pos.setdefault(n, (pos[connected_role][0], pos[connected_role][1] - y_gap))
                else:
                    pos.setdefault(n, (center[0] - 2*x_gap + 60*len(pos), y - y_gap))
            elif a.get("group") == "Education":
                pos.setdefault(n, (center[0] - 2*x_gap + 60*len(pos), y + 2*y_gap))
    return pos

def center_subgraph_positions(pos):
    if not pos:
        return pos
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx = (min(xs)+max(xs))/2
    cy = (min(ys)+max(ys))/2
    return {n: (x - cx, y - cy) for n,(x,y) in pos.items()}

def induced_subgraph(G, nodes):
    node_set = set(nodes)
    edges = [(u,v) for u,v in G.edges() if u in node_set and v in node_set]
    H = nx.DiGraph()
    for n in node_set:
        if n in G.nodes:
            H.add_node(n, **G.nodes[n])
    H.add_edges_from(edges)
    return H

def export_json(G):
    export = {"nodes":[{"id":n, **attrs} for n, attrs in G.nodes(data=True)],
              "edges":[{"source":u,"target":v, **attrs} for u, v, attrs in G.edges(data=True)]}
    return json.dumps(export, indent=2)

def export_png(G):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=400)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=10)
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    buf = io.BytesIO()
    plt.axis('off'); plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight"); plt.close(); buf.seek(0)
    return buf

# ========================= Sidebar Filters & Presets =========================
st.sidebar.header("Filters")
presets = {
    "— Select a preset —": {"companies": [], "skills": [], "start_year": None, "end_year": None, "layout":"auto", "show_years": False},
    "Full journey": {"companies": [], "skills": [], "start_year": None, "end_year": None, "layout":"auto", "show_years": False},
    "Timeline view": {"companies": [], "skills": [], "start_year": None, "end_year": None, "layout":"chronological", "show_years": True},
    "ADP journey": {"companies": ["ADP"], "skills": [], "start_year": 2018, "end_year": None, "layout":"chronological", "show_years": False},
    "AI/ML focus": {"companies": [], "skills": ["AI/ML Integration","GenAI"], "start_year": 2017, "end_year": None, "layout":"chronological", "show_years": False},
    "Data platform & analytics": {"companies": [], "skills": ["Databricks","Lakehouse","Product Analytics","Data Products","Data Enablement"], "start_year": 2018, "end_year": None, "layout":"chronological", "show_years": False},
    "Early career": {"companies": ["AT&T Interactive","Hopeless Records","Best Buy","ExcelFriends.com (Freelance)"], "skills": [], "start_year": 2002, "end_year": 2012, "layout":"chronological", "show_years": False},
}
st.sidebar.subheader("Presets")
preset_choice = st.sidebar.selectbox("Quick filters", options=list(presets.keys()), index=0, key="preset_choice")
if st.sidebar.button("Apply Preset"):
    p = presets.get(st.session_state.get("preset_choice"), presets["— Select a preset —"])
    for k in ("companies","skills","start_year","end_year","search_query","layout","show_year_nodes"):
        if k not in st.session_state:
            st.session_state[k] = [] if k in ("companies","skills") else None
    st.session_state["companies"] = p["companies"]
    st.session_state["skills"] = p["skills"]
    st.session_state["start_year"] = p["start_year"]
    st.session_state["end_year"] = p["end_year"]
    st.session_state["search_query"] = ""
    st.session_state["layout"] = p.get("layout","auto")
    st.session_state["show_year_nodes"] = p.get("show_years", False)

# Search bar (node label contains)
search_query = st.sidebar.text_input("Search nodes (label contains):", value=st.session_state.get("search_query", ""), key="search_query")

show_year_nodes = st.sidebar.checkbox("Show Year Nodes", value=False, key="show_year_nodes")

# Node type filter
node_types = list(GROUP_COLORS.keys())
if "node_types" not in st.session_state:
    st.session_state["node_types"] = node_types
sel_node_types = st.sidebar.multiselect("Node Types to Show", options=node_types, key="node_types")

all_companies = [c["name"] for c in data["companies"]]
sel_companies = st.sidebar.multiselect("Companies", options=all_companies, key="companies")

skills_universe = sorted(set(sum([r["skills"] for r in data["roles"]], [])))
sel_skills = st.sidebar.multiselect("Skills", options=skills_universe, key="skills")

years = list(range(2002, datetime.datetime.today().year + 1))
def _idx(val):
    opts = [None] + years
    return opts.index(val) if val in opts else 0
start_year = st.sidebar.selectbox("Start Year", options=[None] + years, index=_idx(st.session_state.get("start_year")), key="start_year")
end_year = st.sidebar.selectbox("End Year", options=[None] + years, index=_idx(st.session_state.get("end_year")), key="end_year")

# Add color legend to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Node Types")
for node_type, color in GROUP_COLORS.items():
    st.sidebar.markdown(f'<span style="color:{color}">●</span> {node_type}', unsafe_allow_html=True)

filters = {
    "companies": sel_companies,
    "skills": sel_skills,
    "start_year": start_year,
    "end_year": end_year,
    "node_types": sel_node_types
}
layout_mode = st.session_state.get("layout","auto")

# Graph algorithm presets
st.sidebar.subheader("Graph algorithms")
algo = st.sidebar.selectbox("Preset", ["None", "Most central (PageRank)", "Communities (greedy modularity)", "Shortest path: first → latest role"], index=0, key="algo")

# ========================= Build Graph =========================
st.title("Career Knowledge Graph — Edwin Figueroa")
st.caption("Significant nodes are larger (labels stay readable while zoomed out). Presets also re-arrange nodes to tell a story.")

G = build_graph(filters, show_year_nodes=show_year_nodes)

# Node significance → size & label scaling
for n, attrs in G.nodes(data=True):
    score = score_significance(attrs)
    attrs["size"] = 10 + int(score * 8)  # base sizes
    attrs["font"] = {"size": int(10 + score * 3), "vadjust": 0}

# Algorithm presets styling
highlight_edges = set()
if algo == "Most central (PageRank)":
    pr = nx.pagerank(G, alpha=0.85)
    max_pr = max(pr.values()) if pr else 1.0
    for n in G.nodes():
        boost = 30 * (pr[n]/max_pr)
        G.nodes[n]["size"] = G.nodes[n]["size"] + boost
        G.nodes[n]["title"] = (G.nodes[n].get("title","") + f"<br/>PageRank: {pr[n]:.3f}")
elif algo == "Communities (greedy modularity)":
    Ug = G.to_undirected()
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(Ug))
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for i, community in enumerate(comms):
            color = palette[i % len(palette)]
            for n in community:
                G.nodes[n]["color"] = color
                G.nodes[n]["title"] = (G.nodes[n].get("title","") + f"<br/>Community: {i+1}")
    except Exception:
        pass
elif algo == "Shortest path: first → latest role":
    roles = [(n, attrs) for n, attrs in G.nodes(data=True) if attrs.get("group") == "Role"]
    def _year_from_title(t): 
        nums = [int(x) for x in "".join([c if c.isdigit() else " " for c in t]).split() if x.isdigit()]
        return nums[0] if nums else 1900
    if roles:
        first = min(roles, key=lambda x: _year_from_title(x[1].get("title","")))[0]
        latest = max(roles, key=lambda x: _year_from_title(x[1].get("title","")))[0]
        try:
            path = nx.shortest_path(G, first, latest)
            for u, v in zip(path, path[1:]):
                highlight_edges.add((u,v))
        except Exception:
            pass

def render_pyvis(G, layout="auto"):
    net = Network(height="760px", width="100%", bgcolor="#ffffff", font_color="#222", notebook=False, directed=True)
    net.set_options("""
    {
      "interaction": { "hover": true, "tooltipDelay": 120, "zoomView": true, "dragView": true },
      "physics": { 
        "solver": "barnesHut",
        "stabilization": { "enabled": true, "iterations": 150 },
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "nodes": {
        "shape": "dot",
        "scaling": { 
          "min": 10, 
          "max": 80, 
          "label": { 
            "enabled": true, 
            "min": 14, 
            "max": 30, 
            "drawThreshold": 3,
            "maxVisible": 30
          }
        },
        "font": { 
          "size": 16,
          "strokeWidth": 2,
          "strokeColor": "#ffffff"
        }
      },
      "edges": { 
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }, 
        "smooth": { "enabled": true, "type": "dynamic" },
        "font": { "size": 12, "align": "top" },
        "length": 250
      }
    }
    """)
    if layout == "chronological":
        pos = chronological_layout(G)
        pos = center_subgraph_positions(pos)
    else:
        pos = {}

    for nid, attrs in G.nodes(data=True):
        if nid in pos:
            net.add_node(nid,
                         label=attrs.get("label", nid),
                         title=attrs.get("title",""),
                         color=attrs.get("color", None),
                         size=attrs.get("size", 20),
                         font=attrs.get("font", {"size": 14}),
                         x=pos[nid][0], y=pos[nid][1],
                         physics=False, fixed=True)
        else:
            net.add_node(nid,
                         label=attrs.get("label", nid),
                         title=attrs.get("title",""),
                         color=attrs.get("color", None),
                         size=attrs.get("size", 20),
                         font=attrs.get("font", {"size": 14}))

    for src, dst, attrs in G.edges(data=True):
        color = attrs.get("color","#aaaaaa")
        width = 1
        if (src, dst) in highlight_edges:
            color = "#ff0000"; width = 3
        net.add_edge(src, dst, title=attrs.get("label",""), color=color, width=width)
    return net

st.sidebar.markdown("—")
st.sidebar.markdown("**Search & Focus**")
all_node_options = [(n, G.nodes[n].get("label", n)) for n in G.nodes()]
matching = [ (n,l) for n,l in all_node_options if st.session_state.get("search_query","").lower() in l.lower() ] if st.session_state.get("search_query") else []
st.sidebar.caption(f"{len(matching)} match(es)" if matching else "No matches yet")
selected_labels = st.sidebar.multiselect(
    "Select nodes to focus (by label)",
    options=[label for _, label in (matching if st.session_state.get("search_query") else all_node_options)],
    default=[]
)
selected_node_ids = [nid for nid, label in all_node_options if label in selected_labels]

H = induced_subgraph(G, selected_node_ids) if selected_node_ids else G
net = render_pyvis(H, layout=st.session_state.get("layout","auto"))
graph_html = net.generate_html()
st.components.v1.html(graph_html, height=760, scrolling=True)

# ========================= Story Panel =========================
st.subheader("Career Story (auto-generated)")
def pretty_year(y): return "Present" if y is None else str(y)
def role_passes_filters_for_story(r, filters):
    comp_name = next(c["name"] for c in data["companies"] if c["id"]==r["company_id"])
    if filters["companies"] and comp_name not in filters["companies"]:
        return False
    rs, re = role_year_span(r)
    if filters["start_year"] and rs and rs < filters["start_year"]:
        if not (re and re >= filters["start_year"]):
            return False
    if filters["end_year"] and re and re > filters["end_year"]:
        if not (rs and rs <= filters["end_year"]):
            return False
    if filters["skills"] and not any(s in r["skills"] for s in filters["skills"]):
        return False
    return True
roles = sorted([r for r in data["roles"] if role_passes_filters_for_story(r, filters)],
               key=lambda r: parse_date(r["start"]) or datetime.datetime(1900,1,1))
narrative = []
for r in roles:
    comp_name = next(c["name"] for c in data["companies"] if c["id"]==r["company_id"])
    rs, re = role_year_span(r)
    narrative.append(f"From {pretty_year(rs)} to {pretty_year(re)}, served as **{r['title']}** at **{comp_name}**.")
    if r["highlights"]:
        narrative.append("  - " + "  - ".join(r["highlights"][:3]))
if not narrative:
    st.write("Adjust filters or search to see a focused story.")
else:
    st.markdown("\n".join(narrative))

# ========================= Export Section =========================
st.subheader("Export Current View")
col1, col2 = st.columns(2)
with col1:
    st.write("**Export subgraph as JSON**")
    json_str = export_json(H)
    st.download_button("Download JSON", data=json_str, file_name="edwin_career_subgraph.json", mime="application/json")
with col2:
    st.write("**Export subgraph as PNG**")
    png_bytes = export_png(H)
    st.download_button("Download PNG", data=png_bytes, file_name="edwin_career_subgraph.png", mime="image/png")

# ========================= Table =========================
st.subheader("Roles Table")
table = []
for r in data["roles"]:
    comp_name = next(c["name"] for c in data["companies"] if c["id"]==r["company_id"])
    rs, re = role_year_span(r)
    table.append({
        "Company": comp_name,
        "Title": r["title"],
        "Start": rs,
        "End": re if re else "Present",
        "Key Skills": ", ".join(r["skills"]),
        "Top Highlights": "; ".join(r["highlights"][:2])
    })
df = pd.DataFrame(table).sort_values(by="Start")
st.dataframe(df, use_container_width=True)

st.info("Presets rearrange the graph. Significant nodes are larger with labels that stay readable even when zoomed out. Try algorithm presets to surface structure (PageRank, communities, shortest paths).")
