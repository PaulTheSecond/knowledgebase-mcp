import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: 'repositories'
  },
  {
    path: 'repositories',
    loadComponent: () => import('./features/repositories/repositories.component').then(m => m.RepositoriesComponent)
  },
  {
    path: 'queue',
    loadComponent: () => import('./features/queue/queue.component').then(m => m.QueueComponent)
  },
  {
    path: 'chat',
    loadComponent: () => import('./features/chat/chat.component').then(m => m.ChatComponent)
  },
  {
    path: '**',
    redirectTo: 'repositories'
  }
];
