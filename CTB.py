import random
import sys
import numpy as np
import pygame as pg
from pygame.locals import *

#setup display
display_width = 800
display_height = 400
#setup colors
RED = (0,139,139)
GREEN = (0,128,0)
WHITE = (255,255,255)
#specify circle properties
circle_centre_x = 400
circle_centre_y = 50
circle_radius = 20
circle_y_step_falling = 40
#specify rectangle properties
rectangle_left = 400
rectangle_top = 350
rectangle_width = 200
rectangle_height = 50
#set hyperparameters
lr = 0.85
y = 0.99
#set constants
score = 0
missed = 0
reward = 0

class State:
    def __init__(self,rect,Circle):
        self.rect = rect
        self.Circle = Circle

class Circle:
    def __init__(self,circle_centre_x,circle_centre_y):
        self.circle_centre_x = circle_centre_x
        self.circle_centre_y = circle_centre_y

def score_missed_count():
    font = pg.font.SysFont(None,25)
    text = font.render('Score:' + str(score),True,(238,58,140))
    text1 = font.render('Missed:' + str(missed),True,(238,58,140))
    gameDisplay.blit(text,(display_width - 120,10))
    gameDisplay.blit(text1,(display_width - 280,10))

def calculate_score(rect,circle):
    if rect.left <= circle.circle_centre_x <= rect.right:
        return 1
    else:
        return -1

def circle_falling(circle_radius):
    new_x = 100 - circle_radius
    multiplier = random.randint(1,8)
    new_x *= multiplier
    return new_x

rct = pg.Rect(rectangle_left,rectangle_top,rectangle_width,rectangle_height)
Q_Dic = {}
Q = np.zeros([300,3])

def state_to_numbers(s):
    r = s.rect.left #you can also use s.rect.right
    cx = s.Circle.circle_centre_x #circle x position
    cy = s.Circle.circle_centre_y
    n = int(str(r) + str(cx) + str(cy))

    if n in Q_Dic:
        return Q_Dic[n]
    else:
        if len(Q_Dic):
            maximum = max(Q_Dic,key = Q_Dic.get)
            Q_Dic[n] = Q_Dic[maximum] + 1
        else:
            Q_Dic[n] = 1
    return Q_Dic[n]

def get_best_action(s):
    return np.argmax(Q[state_to_numbers(s),:])

def new_state_after_action(s,act):
    if act == 2:
        if s.rect.right + s.rect.width > display_width:
            rct = s.rect
        else:
            rct = pg.Rect(s.rect.left + s.rect.width,s.rect.top,s.rect.width,s.rect.height)

    elif act == 1:
        if s.rect.left - s.rect.width < 0:
            rct  = s.rect
        else:
            rct = pg.Rect(s.rect.left - s.rect.width,s.rect.top,s.rect.width,s.rect.height)

    else:
        rct = s.rect

    new_circle = Circle(s.Circle.circle_centre_x,s.Circle.circle_centre_y+circle_y_step_falling)

    return State(rct,new_circle)

def new_rect_after_action(rect,act):
        if act == 2:
            if s.rect.right + s.rect.width > display_width:
                return rect
            else:
                return pg.Rect(s.rect.left + s.rect.width,s.rect.top,s.rect.width,s.rect.height)

        elif act == 1:
            if s.rect.left - s.rect.width < 0:
                return rect
            else:
                return pg.Rect(s.rect.left - s.rect.width,s.rect.top,s.rect.width,s.rect.height)

        else:
            return rect

#initialize pygame
FPS = 50
clock = pg.time.Clock()
pg.init()
gameDisplay = pg.display.set_mode((display_width,display_height))
pg.display.set_caption('Catch The Ball')

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            sys.exit()
    gameDisplay.fill(WHITE)
    pg.draw.rect(gameDisplay,GREEN,rct)
    pg.draw.circle(gameDisplay,RED,(circle_centre_x,circle_centre_y),circle_radius)

    if circle_centre_y == display_height - rectangle_height - circle_radius:
        reward = calculate_score(rct,Circle(circle_centre_x,circle_centre_y))
        circle_centre_x = circle_falling(circle_radius)
        circle_centre_y = 50

    else:
        reward = 0
        circle_centre_y += circle_y_step_falling

    s = State(rct,Circle(circle_centre_x,circle_centre_y))
    act = get_best_action(s)
    r0 = calculate_score(s.rect,s.Circle)
    s1 = new_state_after_action(s,act)

    Q[state_to_numbers(s),act] += lr*(r0 + y*np.max(Q[state_to_numbers(s1),:]) - Q[state_to_numbers(s),act])
    rct = new_rect_after_action(s.rect,act)
    #put the position of circle where it were
    circle_centre_x = s.Circle.circle_centre_x
    circle_centre_y = s.Circle.circle_centre_y
    score_missed_count()
    if reward == 1:
        score += reward
    elif reward == -1:
        missed += reward
    pg.display.update()
    clock.tick(FPS)
