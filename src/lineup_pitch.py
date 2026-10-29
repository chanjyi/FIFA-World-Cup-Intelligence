"""Historical Starting XI schematic pitch: illustrative positions only.

Actual formations/positions are not systematically provided in OpenFootball.
Never label the schematic arrangement as a verified tactical formation.
"""
import numpy as np
import plotly.graph_objects as go

CYAN = '#18E0FF'
GOLD = '#FFC857'
WHITE = '#D8F8FF'


def pitch_figure(players, team: str, color: str = CYAN, layout: str = '4-3-3', height: int = 450) -> go.Figure:
    players = players.sort_values('order').head(11)
    if layout == '4-4-2':
        groups = [1, 4, 4, 2]
    else:
        groups = [1, 4, 3, 3]
    ypos = [0.115, 0.325, 0.57, 0.82]
    labels = ['Starter 1', 'Starters 2–5', 'Starters 6–8', 'Starters 9–11']
    fig = go.Figure()
    # pitch surface and stripes
    fig.add_shape(type='rect', x0=0, y0=0, x1=1, y1=1, fillcolor='#06483F', line=dict(color='#67E6D4', width=2), layer='below')
    for i in range(8):
        if i % 2 == 0:
            fig.add_shape(type='rect', x0=0, x1=1, y0=i/8, y1=(i+1)/8,
                          fillcolor='rgba(17,100,85,.38)', line=dict(width=0), layer='below')
    pitch_line=dict(color='rgba(210,255,244,.68)',width=1.6)
    fig.add_shape(type='line', x0=0, x1=1, y0=.5,y1=.5,line=pitch_line)
    fig.add_shape(type='circle', x0=.39,x1=.61,y0=.435,y1=.565,line=pitch_line)
    fig.add_shape(type='circle', x0=.496,x1=.504,y0=.496,y1=.504,fillcolor=WHITE,line=dict(width=0))
    fig.add_shape(type='rect', x0=.24,x1=.76,y0=0,y1=.17,line=pitch_line,fillcolor='rgba(0,0,0,0)')
    fig.add_shape(type='rect', x0=.24,x1=.76,y0=.83,y1=1,line=pitch_line,fillcolor='rgba(0,0,0,0)')
    fig.add_shape(type='rect', x0=.38,x1=.62,y0=0,y1=.065,line=pitch_line,fillcolor='rgba(0,0,0,0)')
    fig.add_shape(type='rect', x0=.38,x1=.62,y0=.935,y1=1,line=pitch_line,fillcolor='rgba(0,0,0,0)')
    point_rows=[]
    counter=0
    for size,y in zip(groups,ypos):
        xs=np.linspace(.16,.84,size) if size>1 else [0.5]
        for x in xs:
            if counter>=len(players):break
            p=players.iloc[counter]
            point_rows.append((x,y,int(p['order']),str(p['player']),bool(p['captain'])))
            counter+=1
    if point_rows:
        xs,ys,nums,names,caps=zip(*point_rows)
        fig.add_trace(go.Scatter(x=xs,y=ys,mode='markers+text',
                                 marker=dict(size=32,color=color,line=dict(color='#CFF7FF',width=2)),
                                 text=[str(n) for n in nums],textfont=dict(color='#031320',size=12,family='Arial Black'),
                                 textposition='middle center',
                                 customdata=[[n, bool(c)] for n,c in zip(names,caps)],
                                 hovertemplate='<b>%{customdata[0]}</b><br>Source order #%{text}<extra></extra>',
                                 showlegend=False))
    fig.update_xaxes(range=[-.03,1.03],visible=False,fixedrange=True)
    fig.update_yaxes(range=[-.03,1.03],visible=False,fixedrange=True,scaleanchor='x',scaleratio=1)
    fig.update_layout(height=height,margin=dict(l=0,r=0,t=42,b=3),
                      title=dict(text=f'{team} • Starting XI',font=dict(color=WHITE,size=16),x=.5),
                      paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      hoverlabel=dict(bgcolor='#071D34',font=dict(color=WHITE)))
    return fig
