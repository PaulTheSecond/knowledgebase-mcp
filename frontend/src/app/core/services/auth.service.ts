import { Injectable, signal } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private readonly STORAGE_KEY = 'kb_auth_credentials';
  private readonly USER_KEY = 'kb_auth_username';

  readonly isAuthenticated = signal<boolean>(false);
  readonly isAuthModalOpen = signal<boolean>(false);
  readonly currentUser = signal<string>('admin');

  constructor() {
    const savedCreds = localStorage.getItem(this.STORAGE_KEY);
    const savedUser = localStorage.getItem(this.USER_KEY);
    if (savedCreds) {
      this.isAuthenticated.set(true);
      if (savedUser) this.currentUser.set(savedUser);
    } else {
      // По умолчанию предзаполняем или предлагаем авторизацию
      this.isAuthModalOpen.set(true);
    }
  }

  getAuthHeader(): string | null {
    const creds = localStorage.getItem(this.STORAGE_KEY);
    return creds ? `Basic ${creds}` : null;
  }

  setCredentials(username: string, pass: string): void {
    const encoded = btoa(`${username}:${pass}`);
    localStorage.setItem(this.STORAGE_KEY, encoded);
    localStorage.setItem(this.USER_KEY, username);
    this.currentUser.set(username);
    this.isAuthenticated.set(true);
    this.isAuthModalOpen.set(false);
  }

  logout(): void {
    localStorage.removeItem(this.STORAGE_KEY);
    localStorage.removeItem(this.USER_KEY);
    this.isAuthenticated.set(false);
    this.isAuthModalOpen.set(true);
  }

  promptLogin(): void {
    this.isAuthenticated.set(false);
    this.isAuthModalOpen.set(true);
  }

  closeModal(): void {
    if (this.isAuthenticated()) {
      this.isAuthModalOpen.set(false);
    }
  }
}
